from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_sync_groups: List[dist.ProcessGroup] = []
_q: Optional[queue.Queue[str]] = None
_worker_thread: Optional[threading.Thread] = None
_scheduler_thread: Optional[threading.Thread] = None


def rank_consensus(func=None, *, same_params=None, same_results=None, **kwargs):

    if kwargs:
        raise TypeError(
            f"rank_consensus() got unexpected keyword argument(s): " f"{list(kwargs)}"
        )

    params_selector = _normalize_selector(same_params, "same_params")
    results_selector = _normalize_selector(same_results, "same_results")

    def decorator(func: Callable) -> Callable:


        if not envs.SGLANG_ENABLE_RANK_CONSENSUS_CHECKER.get():
            return func


        if isinstance(func, (classmethod, staticmethod)):
            raw_func = func.__func__
            descriptor_type = type(func)
        else:
            raw_func = func
            descriptor_type = None
        sig = inspect.signature(raw_func)


        skip_name: Optional[str] = None
        if _is_method_with_receiver(func) and len(sig.parameters) > 0:
            skip_name = next(iter(sig.parameters))

        @functools.wraps(raw_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            params_payload = "<no check>"
            if params_selector is not None:


                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                params_payload = _build_payload(
                    "call", params_selector, arguments, skip_name
                )
            assert_same("%s called params=%s", raw_func.__name__, params_payload)

            result = raw_func(*args, **kwargs)

            result_payload = "<no check>"
            if results_selector is not None:
                result_scope = {"result": result}
                result_payload = _build_payload(
                    "return",
                    results_selector,
                    result_scope,
                )
            assert_same("%s returns result=%s", raw_func.__name__, result_payload)
            return result


        if descriptor_type is staticmethod:
            return staticmethod(wrapper)
        if descriptor_type is classmethod:
            return classmethod(wrapper)
        return wrapper

    if func is not None:

        return decorator(func)
    else:

        return decorator


def _normalize_selector(
    value: None | bool | str | list[str], name: str
) -> None | bool | list[str]:

    if value is None or value is False:
        return None
    if value is True:
        return True
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(s, str) for s in value):
        return list(value)
    raise TypeError(f"{name} must be True / False / str / list[str], got {value!r}")


def _is_method_with_receiver(func: Any) -> bool:

    if isinstance(func, staticmethod):
        return False
    if isinstance(func, classmethod):
        return True
    if inspect.isfunction(func):
        qualname = getattr(func, "__qualname__", "")


        if "." in qualname and "<locals>" not in qualname:
            return True
    return False


def _build_payload(
    tag: str,
    selector: bool | list[str],
    scope: dict[str, Any],
    skip_name: Optional[str] = None,
) -> str:

    if selector is True:


        if tag == "call":
            if skip_name is not None:
                scope = {k: v for k, v in scope.items() if k != skip_name}
            return repr(scope)
        return repr(scope["result"])
    parts: list[str] = []
    for expr in selector:
        value = _eval_selector(expr, scope)
        parts.append(f"{expr}={value!r}")
    return " | ".join(parts)


def _eval_selector(expr: str, scope: dict[str, Any]) -> Any:

    safe_builtins = {
        "len": len,
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "set": set,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
    }
    return eval(expr, {"__builtins__": safe_builtins}, dict(scope))


def enabled() -> bool:

    return _q is not None


def assert_same(msg_fmt: str, *args: Any) -> None:

    if not enabled():
        return


    if threading.current_thread() is not _scheduler_thread:
        raise RuntimeError("rdc.assert_same must be called from the scheduler thread")


    _q.put(msg_fmt % args)


def configure(groups: List[GroupCoordinator]) -> None:

    global _sync_groups, _q, _worker_thread, _scheduler_thread

    if not envs.SGLANG_ENABLE_RANK_CONSENSUS_CHECKER.get():
        return

    logger.warning(
        "Rank consensus checker is enabled. The server will suicide if rank divergence detected."
    )


    _sync_groups = _create_sync_groups(groups)
    _q = queue.Queue()


    _scheduler_thread = threading.current_thread()
    _worker_thread = threading.Thread(
        target=_worker_loop, name="rank_consensus_checker", daemon=True
    )
    _worker_thread.start()


def _create_sync_groups(
    groups: List[GroupCoordinator],
) -> List[dist.ProcessGroup]:

    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    dedicated: List[dist.ProcessGroup] = []
    seen_rank_sets: set[tuple[int, ...]] = set()
    for group in groups:
        if group is None:
            continue

        if torch.distributed.get_world_size(group=group.cpu_group) == 1:
            continue
        group_ranks = tuple(torch.distributed.get_process_group_ranks(group.cpu_group))
        if group_ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(group_ranks)
        pg = create_custom_parallel_group(group_ranks=list(group_ranks), backend="gloo")
        if pg is not None:
            dedicated.append(pg)
    return dedicated


def _destroy_dedicated_groups() -> None:
    for pg in _sync_groups:
        try:
            torch.distributed.destroy_process_group(pg)
        except Exception:
            pass


def shutdown() -> None:

    global _q, _worker_thread, _sync_groups, _scheduler_thread

    q = _q
    if q is None:
        return


    q.put(None)
    if _worker_thread is not None:
        _worker_thread.join()
        _worker_thread = None


    _destroy_dedicated_groups()
    _q = None
    _sync_groups = []
    _scheduler_thread = None


def _worker_loop() -> None:

    while _q is not None:

        first = _q.get()
        if first is None:

            return


        count = _all_reduce_min_int(_q.qsize())
        events: List[str] = [first]
        shutdown_signaled = False
        for _ in range(count):
            event = _q.get()
            if event is None:


                shutdown_signaled = True
                break
            events.append(event)


        _check_for_consensus(events)

        if shutdown_signaled:
            return


def _all_reduce_min_int(value: int) -> int:

    tensor = torch.tensor([value], dtype=torch.int64)
    for group in _sync_groups:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return int(tensor.item())


def _check_for_consensus(events: list[str]) -> None:

    hasher = hashlib.sha1()
    for msg in events:
        hasher.update(msg.encode("utf-8"))


    value_bytes = hasher.digest()
    min_value = torch.tensor(list(hasher.digest()), dtype=torch.uint8)
    max_value = min_value.clone()
    for group in _sync_groups:
        dist.all_reduce(min_value, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(min_value, max_value):

        logger.critical(
            f"Found rank divergence for {len(events)} events(s)! local hash: {value_bytes.hex()}, events = {events}"
        )
        for handler in logger.handlers:
            handler.flush()


        os._exit(1)

    logger.debug(f"Consensus check passed for {len(events)} event(s).")
