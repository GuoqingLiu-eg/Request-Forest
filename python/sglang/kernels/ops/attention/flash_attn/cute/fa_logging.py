# Copyright (c) 2025, Tri Dao.


import logging
import os
import sys

import cutlass.cute as cute
from cutlass import const_expr

_LOG_LEVEL_NAMES = {"off": 0, "host": 1, "kernel": 2, "max": 3}


def _parse_log_level(raw: str) -> int:
    if raw in _LOG_LEVEL_NAMES:
        return _LOG_LEVEL_NAMES[raw]
    try:
        level = int(raw)
    except ValueError:
        return 0
    return max(0, min(level, 3))


_fa_log_level: int = _parse_log_level(os.environ.get("FA_LOG_LEVEL", "0"))

_logger = logging.getLogger("flash_attn")
_logger.addHandler(logging.NullHandler())
_default_handler: logging.Handler | None = None


def _configure_default_handler() -> None:
    global _default_handler
    if _fa_log_level >= 1:
        if _default_handler is None:
            _default_handler = logging.StreamHandler(sys.stdout)
            _default_handler.setFormatter(logging.Formatter("[FA] %(message)s"))
            _logger.addHandler(_default_handler)
        _logger.setLevel(logging.DEBUG)
    else:
        if _default_handler is not None:
            _logger.removeHandler(_default_handler)
            _default_handler = None
        _logger.setLevel(logging.WARNING)


_configure_default_handler()


def get_fa_log_level() -> int:
    return _fa_log_level


def set_fa_log_level(level: int | str) -> None:

    global _fa_log_level
    if isinstance(level, str):
        level = _parse_log_level(level)
    _fa_log_level = max(0, min(int(level), 3))
    _configure_default_handler()


def fa_log(level: int, msg: str):
    if _fa_log_level >= level:
        _logger.info(msg)


def fa_printf(level: int, fmt, *args):
    if const_expr(_fa_log_level >= level):
        cute.printf(fmt, *args)
