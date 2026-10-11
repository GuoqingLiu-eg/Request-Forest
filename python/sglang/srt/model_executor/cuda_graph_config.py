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


import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Phase:


    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)


class Backend:


    FULL = "full"
    BREAKABLE = "breakable"
    TC_PIECEWISE = "tc_piecewise"
    DISABLED = "disabled"
    ALL = (FULL, BREAKABLE, TC_PIECEWISE, DISABLED)


ALLOWED_BACKENDS_PER_PHASE = {
    Phase.DECODE: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),


    Phase.PREFILL: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TC_PIECEWISE,
        Backend.DISABLED,
    ),
}


ALLOWED_KEYS_PER_PHASE = {
    Phase.DECODE: ("backend", "max_bs", "bs", "tc_compiler"),
    Phase.PREFILL: (
        "backend",
        "max_bs",
        "bs",
        "tc_compiler",
        "full_prefill_max_req",
        "full_prefill_prefix_chunk_tokens",
    ),
}


@dataclass
class PhaseConfig:


    backend: str = Backend.DISABLED
    max_bs: Optional[int] = None
    bs: Optional[List[int]] = None

    tc_compiler: str = "eager"


    full_prefill_max_req: Optional[int] = None


    full_prefill_prefix_chunk_tokens: Optional[int] = None


def default_prefill_backend() -> str:

    from sglang.srt.utils import is_cuda

    return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE


@dataclass
class CudaGraphConfig:


    decode: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=Backend.FULL)
    )
    prefill: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(backend=default_prefill_backend())
    )

    def __getitem__(self, phase: str) -> PhaseConfig:

        if phase not in Phase.ALL:
            raise KeyError(phase)
        return getattr(self, phase)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:


        baseline = default_cuda_graph_config()
        return {
            Phase.DECODE: _diff_phase(self.decode, baseline.decode),
            Phase.PREFILL: _diff_phase(self.prefill, baseline.prefill),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Dict[str, Any]]]) -> "CudaGraphConfig":

        cfg = cls()
        if not raw:
            return cfg
        for phase, phase_settings in raw.items():
            if phase not in Phase.ALL or not isinstance(phase_settings, dict):
                continue
            phase_cfg = getattr(cfg, phase)
            allowed = ALLOWED_KEYS_PER_PHASE[phase]
            for key, value in phase_settings.items():
                if key in allowed:
                    setattr(phase_cfg, key, value)
        return cfg


def default_cuda_graph_config() -> CudaGraphConfig:

    return CudaGraphConfig()


def _diff_phase(actual: PhaseConfig, baseline: PhaseConfig) -> Dict[str, Any]:

    return {
        f.name: getattr(actual, f.name)
        for f in dataclasses.fields(actual)
        if getattr(actual, f.name) != getattr(baseline, f.name)
    }


def check_cuda_graph_backend(phase: str, backend: str) -> bool:

    from sglang.srt.runtime_context import get_exec

    try:
        cfg = get_exec().graph.cuda_graph_config
    except ValueError:
        return False
    if cfg is None or phase not in Phase.ALL:
        return False
    return getattr(cfg, phase).backend == backend


def cuda_graph_fully_disabled() -> bool:

    return check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ) and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)


def parse_cuda_graph_config_arg(raw: str) -> Dict[str, Dict[str, Any]]:

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-config must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-config must be a JSON object, got {type(parsed).__name__}"
        )

    result: Dict[str, Dict[str, Any]] = {}
    for phase, phase_settings in parsed.items():
        phase = str(phase)
        if phase not in Phase.ALL:
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config: unknown phase '{phase}', expected one of {Phase.ALL}"
            )
        if not isinstance(phase_settings, dict):
            raise argparse.ArgumentTypeError(
                f"--cuda-graph-config['{phase}'] must be a JSON object, got "
                f"{type(phase_settings).__name__}"
            )
        allowed = ALLOWED_KEYS_PER_PHASE[phase]
        result[phase] = {}
        for key, value in phase_settings.items():
            if key not in allowed:
                raise argparse.ArgumentTypeError(
                    f"--cuda-graph-config['{phase}']: unknown key '{key}', expected one of {allowed}"
                )
            result[phase][key] = value
    return result


def explicit_keys_in(
    settings: Optional[Dict[str, Dict[str, Any]]],
) -> set:

    out: set = set()
    if not settings:
        return out
    for phase, phase_settings in settings.items():
        if not isinstance(phase_settings, dict):
            continue
        for key in phase_settings.keys():
            out.add((phase, key))
    return out
