import fcntl
import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_align, ceil_div, get_available_gpu_memory, is_musa

logger = logging.getLogger(__name__)

_is_musa = is_musa()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()


os.environ["DG_JIT_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()


os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:


        _BUILTIN_M_LIST += list(range(1, 1025))


        next_m, sample_step = 1024, 2
        chunked_prefill_size = get_schedule().chunked_prefill_size
        max_prefill_bs = (
            min(chunked_prefill_size, 32 * 1024)
            if chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:

        m_max = 1024 * 16
        chunked_prefill_size = get_schedule().chunked_prefill_size
        if chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif chunked_prefill_size > 8192:
            m_max = chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id


    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_BF16_MASKED = auto()
    GROUPED_GEMM_NT_BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


@contextmanager
def _local_rank_compile_lock(
    kernel_type: DeepGemmKernelType, n: int, k: int, num_groups: int
):

    lock_dir = Path(os.environ["DG_JIT_CACHE_DIR"]) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{kernel_type.name}_n{n}_k{k}_g{num_groups}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True


        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        with _local_rank_compile_lock(kernel_type, n, k, num_groups):
            _compile_deep_gemm_one_type_all(
                kernel_type=kernel_type,
                n=n,
                k=k,
                num_groups=num_groups,
                m_list=_BUILTIN_M_LIST,
            )


def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:


    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))


        memory_budget = get_available_gpu_memory(
            device="cuda", gpu_id=torch.cuda.current_device()
        )


        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:

            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 2048
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]


        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        has_compile_mode_api = hasattr(deep_gemm, "get_compile_mode") and hasattr(
            deep_gemm, "set_compile_mode"
        )
        if has_compile_mode_api:
            old_compile_mode = deep_gemm.get_compile_mode()
            deep_gemm.set_compile_mode(1)


        for m in tqdm(m_list, desc="DeepGEMM warmup"):
            executor.execute(m=m)
        if has_compile_mode_api:
            deep_gemm.set_compile_mode(old_compile_mode)


        torch.cuda.current_stream().synchronize()
        del executor
        torch.cuda.empty_cache()
    finally:

        restore_symmetric_memory_context(saved_context)


class _BaseWarmupExecutor:
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
            DeepGemmKernelType.GEMM_NT_BF16BF16F32: _BF16F32WarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG: _BF16GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED: _BF16GroupedMaskedWarmupExecutor,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:

        _GB = 1 << 30
        if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
            return (max_m * k + n * k + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG:
            return (
                max_m * k * 2 + num_groups * n * k * 2 + max_m * 4 + max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:

            return (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED:
            return (
                num_groups * max_m * k * 2
                + num_groups * n * k * 2
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _NormalWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            self.m_indices[:m],
        )


class _BF16GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            self.a[:m],
            self.b,
            self.out[:m],
            self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,

            expected_m=m,
        )


class _BF16F32WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)

    def execute(self, m):
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])


class _BF16GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.a = torch.empty(
            (num_groups, max_m, k), device="cuda", dtype=torch.bfloat16
        )
        self.b = torch.empty((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            self.a,
            self.b,
            self.out,
            masked_m=self.masked_m,

            expected_m=m,
        )


def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if _is_musa:
        return nullcontext()

    return _deep_gemm_execution_hook(m, n, k, num_groups, kernel_type)


@contextmanager
def _deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield


def pp_parallel_deep_gemm_warmup(runner) -> None:

    model_runner = runner.model_runner


    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.utils.common import require_mlp_sync

    n_sms = torch.cuda.get_device_properties(model_runner.device).multi_processor_count
    block_m = 64
    cp = max(get_cp_padding_align_size(), 1)

    attn_tp_size = get_parallel().attn_tp_size
    mlp_sync = require_mlp_sync()

    def _align(bs: int) -> int:


        align = cp
        if mlp_sync and attn_tp_size > 1:
            align = math.lcm(cp, attn_tp_size)
        return ceil_align(bs, align)

    batch_sizes = sorted(
        {
            _align(bs)
            for bs in (
                1,
                2 * block_m,
                max(n_sms // 8, 2) * block_m,
                max(n_sms // 4, 4) * block_m,
                n_sms * block_m,
            )
        }
    )


    disagg_mode = get_disagg().disaggregation_mode
    run_decode = model_runner.is_generation and disagg_mode != "prefill"
    run_extend = disagg_mode != "decode"

    logger.info(
        "PP-parallel DeepGEMM warmup start "
        "(pp_rank=%d, tp_rank=%d, batch_sizes=%s, disagg=%s).",
        model_runner.ps.pp_rank,
        model_runner.ps.tp_rank,
        batch_sizes,
        disagg_mode,
    )


    dummy_buffers = runner._alloc_dummy_decode_buffers(max(batch_sizes))

    t0 = time.perf_counter()
    with torch.inference_mode():
        for bs in batch_sizes:
            if run_decode:
                runner._dummy_run(
                    batch_size=bs,
                    forward_mode_override=ForwardMode.DECODE,
                    buffers=dummy_buffers,
                )
            if run_extend:
                runner._dummy_run(
                    batch_size=bs,
                    forward_mode_override=ForwardMode.EXTEND,
                    buffers=dummy_buffers,
                )

    logger.info(
        "PP-parallel DeepGEMM warmup done in %.2fs (pp_rank=%d).",
        time.perf_counter() - t0,
        model_runner.ps.pp_rank,
    )
