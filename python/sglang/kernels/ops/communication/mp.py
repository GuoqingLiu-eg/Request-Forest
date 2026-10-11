from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
from typing import Any, Callable, List, NoReturn, Optional, Sequence

import psutil
import torch

logger = logging.getLogger(__name__)


def register_comm_cleanup(comm: Any) -> None:


    def _safe_close() -> None:
        try:
            comm.close()
        except Exception:
            pass

        try:
            comm.disabled = True
        except Exception:
            pass

        try:
            delattr(comm, "obj")
        except Exception:
            pass

        try:
            comm._ptr = 0
        except Exception:
            pass

    atexit.register(_safe_close)


def _kill_pgroup(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _kill_descendants(pid: int) -> None:

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    descendants = root.children(recursive=True)
    for proc in descendants:
        try:
            proc.kill()
        except psutil.Error:

            pass
    psutil.wait_procs(descendants, timeout=5)


def _extract_num_gpus_override(
    argv: list[str],
) -> tuple[list[int] | None, list[str]]:

    override: list[int] = []
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--num-gpu", "--num-gpus"):
            if i + 1 >= len(argv):
                raise ValueError(f"missing value for {a} (expected e.g. `{a} 2,4`)")
            override.extend(int(x) for x in argv[i + 1].split(","))
            i += 2
        elif a.startswith("--num-gpu=") or a.startswith("--num-gpus="):
            _, val = a.split("=", 1)
            override.extend(int(x) for x in val.split(","))
            i += 1
        else:
            remaining.append(a)
            i += 1
    return (override if override else None), remaining


def multigpu_launch(
    name: str,
    file: str,
    num_gpus: Sequence[int],
    env_key: str,
    inner: Callable[[], int],
    kind: str,
    pre_launch_fn: Optional[Callable[[List[int]], None]] = None,
    timeout: Optional[int] = None,
) -> NoReturn | None:

    pid_key = env_key + "_PID"
    if env_key in os.environ:
        assert pid_key in os.environ
        if name != "__main__":
            return
        rank = int(os.environ["LOCAL_RANK"])
        if rank != 0:
            sys.stdout = open(os.devnull, "w")
        torch.cuda.set_device(rank)
        return sys.exit(inner())
    assert pid_key not in os.environ
    if name != "__main__":
        return logger.warning(
            f"{file} can not directly run with `pytest`. "
            "Use `python` to invoke it, which will internally relaunch it "
            "under torchrun for each requested number of GPUs."
        )
    num_devices = torch.cuda.device_count()
    override, forwarded_args = _extract_num_gpus_override(sys.argv[1:])
    if override is not None:
        logger.info(f"--num-gpu override: running only with {override}")
        num_gpus = override
        for N in num_gpus:
            if N <= 1 or N > num_devices:
                raise ValueError(
                    f"Invalid number of GPUs requested: {N} "
                    f"(available: {num_devices})"
                )
    os.environ[env_key] = "1"
    os.environ[pid_key] = str(os.getpid())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    signal.signal(signal.SIGINT, signal.default_int_handler)
    runnable: List[int] = []
    for N in sorted(num_gpus):
        assert N > 1
        if N > num_devices:
            logger.warning(f"Skipping {kind} with {N} GPUs ({num_devices} available)")
            continue
        runnable.append(N)
    if pre_launch_fn is not None and runnable:
        logger.info(f"Running pre-launch hook for world sizes {runnable}")
        pre_launch_fn(runnable)
    for N in runnable:
        logger.info(f"Running {kind} with {N} GPUs")
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(N),
            "--local-addr",
            "127.0.0.1",
            file,
        ]
        cmd += forwarded_args
        proc = subprocess.Popen(cmd, start_new_session=True)
        pgid = proc.pid
        returncode = -1
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_descendants(os.getpid())
            _kill_pgroup(pgid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if timed_out:
            logger.error(f"{kind} (nproc={N}) timed out after {timeout} seconds")
            sys.exit(1)
        if returncode != 0:
            logger.error(f"{kind} failed with {N} GPUs (exit {returncode})")
            sys.exit(returncode)
    logger.info(f"All {kind}s passed")
