from typing import Optional

_PRIORITY_MIN = 0
_PRIORITY_MAX = 31
_LOW_PRIORITY_VALUE = "LOW"
_HIGH_PRIORITY_VALUE = "HIGH"

UNKNOWN_PRIORITY_VALUE = "UNKNOWN"


def transform_priority(priority: Optional[int]) -> str:

    if priority is None:
        return UNKNOWN_PRIORITY_VALUE
    elif priority < _PRIORITY_MIN:
        return _LOW_PRIORITY_VALUE
    elif priority >= _PRIORITY_MAX:
        return _HIGH_PRIORITY_VALUE
    else:
        return str(priority)
