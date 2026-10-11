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
import re
from dataclasses import replace as _dataclass_replace

import torch
import torch.fx.graph as fx_graph

_DEVICE_REPR_RE = re.compile(r"\bdevice\(type='([^']+)'(?:,\s*index=(\d+))?\)")


def _replace_device_repr(m: re.Match) -> str:
    dev_type = m.group(1)
    dev_index = m.group(2)
    if dev_index is not None:
        return f"torch.device('{dev_type}:{dev_index}')"
    return f"torch.device('{dev_type}')"


def patch_fx_custom_device() -> None:

    original = fx_graph.CodeGen._gen_python_code

    def patched(self, nodes, root_module, namespace, **kwargs):
        result = original(self, nodes, root_module, namespace, **kwargs)
        new_src = _DEVICE_REPR_RE.sub(_replace_device_repr, result.src)
        if new_src is result.src:
            return result
        result.globals.setdefault("torch", torch)
        if hasattr(result, "_replace"):
            return result._replace(src=new_src)
        return _dataclass_replace(result, src=new_src)

    fx_graph.CodeGen._gen_python_code = patched
