import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger(__name__)

enable_startup_metrics = False
STARTUP_LATENCY_SECONDS = None

_max_durations: Dict[str, float] = {}


def enable_startup_timer():


    from prometheus_client import Gauge

    global enable_startup_metrics, STARTUP_LATENCY_SECONDS
    enable_startup_metrics = True

    STARTUP_LATENCY_SECONDS = Gauge(
        "sglang:startup_latency_breakdown_seconds_max",
        "Startup latency breakdown in seconds by context, only records the maximum duration if the context is called multiple times.",
        labelnames=["context"],
        multiprocess_mode="mostrecent",
    )


def set_startup_metric(context: str, value: float, should_log: bool = True):

    if should_log:
        logger.info(f"Setting startup metric: {context} took {value:.3f}s")

    if not enable_startup_metrics:
        return
    current_max = _max_durations.get(context, 0.0)
    if value > current_max:
        _max_durations[context] = value
        STARTUP_LATENCY_SECONDS.labels(context=context).set(value)


def reset_startup_timers():

    global _max_durations
    _max_durations.clear()


def get_max_duration(context: str) -> Optional[float]:

    return _max_durations.get(context)


@contextmanager
def startup_timer(name: str, log_only: bool = False) -> Generator[None, None, None]:

    start_time = time.monotonic()
    try:
        yield
    finally:
        duration_seconds = time.monotonic() - start_time


        current_max = _max_durations.get(name, 0.0)
        is_new_max = duration_seconds > current_max

        if is_new_max:
            _max_durations[name] = duration_seconds


            if enable_startup_metrics and not log_only:
                STARTUP_LATENCY_SECONDS.labels(context=name).set(duration_seconds)


        logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")


def time_startup_latency(
    func: Callable = None, name: Optional[str] = None, log_only: bool = False
) -> Callable[..., Any]:


    def measure(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal name
        name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_seconds = time.monotonic() - start_time


                current_max = _max_durations.get(name, 0.0)
                is_new_max = duration_seconds > current_max

                if is_new_max:
                    _max_durations[name] = duration_seconds


                    if enable_startup_metrics and not log_only:
                        STARTUP_LATENCY_SECONDS.labels(context=name).set(
                            duration_seconds
                        )


                logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")

        return wrapper

    if func:
        return measure(func)
    else:
        return measure
