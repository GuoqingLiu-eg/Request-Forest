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


from __future__ import annotations

from array import array
from typing import Any


def validate_list_i64_1d(v: Any) -> list[int]:

    if v is None:
        raise ValueError("must not be None")
    if not isinstance(v, list):
        raise ValueError(f"must be list; got {type(v).__name__}")
    if not v:
        return v
    if not isinstance(v[0], int):
        raise ValueError(f"elements must be int; got {type(v[0]).__name__}")
    try:
        array("q", v)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"contains non-int64 element: {e}") from None
    return v


def validate_optional_list_i64_1d_2d(
    v: Any,
) -> list[int] | list[list[int]] | None:

    if v is None:

        return v
    if not isinstance(v, list):
        raise ValueError(f"must be list or null; got {type(v).__name__}")
    if not v:

        return v
    if isinstance(v[0], int):

        return validate_list_i64_1d(v)
    if isinstance(v[0], list):

        for i, row in enumerate(v):
            try:
                validate_list_i64_1d(row)
            except ValueError as e:
                raise ValueError(f"row {i}: {e}") from None
        return v
    raise ValueError(f"elements must be int or list; got {type(v[0]).__name__}")
