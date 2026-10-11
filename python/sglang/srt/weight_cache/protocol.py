# SPDX-License-Identifier: Apache-2.0


import hashlib
import json
import logging
import os
import pickle
import signal
import struct
from typing import Any, Dict, Optional

import msgspec

from sglang.srt.utils.common import safe_pickle_loads

logger = logging.getLogger(__name__)


WEIGHT_CACHE_SOCKET_TEMPLATE = "/tmp/sglang_weight_cache_rank{global_rank}.sock"


WEIGHT_CACHE_READY_TEMPLATE = "/tmp/sglang_weight_cache_rank{global_rank}.ready"


class CacheConfig(msgspec.Struct):


    model_path: str
    model_arch: str
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    ep_size: int
    moe_dp_size: int
    moe_dp_rank: int
    moe_ep_rank: int
    enable_dp_attention: bool
    enable_dp_lm_head: bool
    attn_cp_size: int
    moe_dense_tp_size: Optional[int]
    moe_a2a_backend: str
    quant_method: str
    quant_config_hash: str
    dtype: str
    revision: str


    device_capability: str
    torch_version: str

    def matches(self, other: "CacheConfig") -> bool:

        return self == other

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__struct_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheConfig":
        return cls(**d)


def hash_quant_config(quant_config: Any) -> str:

    if quant_config is None:
        return ""
    try:
        if hasattr(quant_config, "to_dict"):
            config_str = json.dumps(quant_config.to_dict(), sort_keys=True)
        elif isinstance(quant_config, dict):
            config_str = json.dumps(quant_config, sort_keys=True)
        elif hasattr(quant_config, "__dict__"):
            config_str = (
                type(quant_config).__name__
                + ":"
                + json.dumps(
                    {
                        k: v
                        for k, v in sorted(quant_config.__dict__.items())
                        if not k.startswith("_")
                        and isinstance(
                            v, (str, int, float, bool, type(None), list, dict)
                        )
                    },
                    sort_keys=True,
                )
            )
        else:
            config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()
    except Exception:
        config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()


def get_quant_method_name(quant_config: Any) -> str:

    if quant_config is None:
        return ""
    if isinstance(quant_config, str):
        return quant_config
    if hasattr(quant_config, "get_name"):
        return quant_config.get_name()
    if hasattr(quant_config, "name"):
        return quant_config.name
    return type(quant_config).__name__


class UnsupportedQuantForIPCError(RuntimeError):
    pass


def _get_quant_field(quant_config: Any, key: str) -> Any:

    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config.get(key)
    return getattr(quant_config, key, None)


def _fp8_round_trips_via_ipc(quant_config: Any) -> bool:

    return _get_quant_field(quant_config, "weight_block_size") is not None


IPC_QUANT_ALLOWLIST = {
    "": lambda _quant_config: True,
    "fp8": _fp8_round_trips_via_ipc,
}


def is_ipc_quant_supported(quant_method: str, quant_config: Any) -> bool:

    predicate = IPC_QUANT_ALLOWLIST.get(quant_method)
    if predicate is None:
        return False
    return bool(predicate(quant_config))


def check_ipc_quant_support(
    quant_method: str, quant_config: Any, *, where: str
) -> None:

    if is_ipc_quant_supported(quant_method, quant_config):
        return
    verified = ", ".join(
        (repr(m) if m else "'' (unquantized)") for m in IPC_QUANT_ALLOWLIST
    )
    raise UnsupportedQuantForIPCError(
        f"[weight_cache:{where}] quantization method {quant_method!r} is not "
        f"verified for CUDA IPC zero-copy weight sharing. Its "
        f"process_weights_after_loading may stamp Python-side metadata "
        f"(e.g. format_ue8m0) or repack/transpose weights into shapes the "
        f"meta-initialized client cannot reproduce, which would silently serve "
        f"wrong-numerics weights. Verified methods: {verified}. Note: FP8 is "
        f"only verified for block-wise configs (weight_block_size set), not "
        f"per-tensor FP8. Disable the weight cache (--weight-cache-mode off) "
        f"for this model."
    )


MAX_MSG_SIZE = 256 * 1024 * 1024


def send_msg(sock, obj: Any) -> None:

    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_msg(sock) -> Any:

    header = _recv_exact(sock, 4)
    if header is None:
        raise ConnectionError("Connection closed while reading message header")
    length = struct.unpack("!I", header)[0]
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message size {length} exceeds {MAX_MSG_SIZE} byte cap")
    data = _recv_exact(sock, length)
    if data is None:
        raise ConnectionError("Connection closed while reading message body")
    return safe_pickle_loads(data)


def _recv_exact(sock, n: int) -> Optional[bytes]:

    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def compute_env_stamp() -> Dict[str, str]:

    device_capability = ""
    torch_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        pass
    try:
        from sglang.srt.platforms import current_platform

        cap = current_platform.get_device_capability()
        if cap is not None:
            device_capability = f"{cap.major}.{cap.minor}"
    except Exception:
        pass
    return {"device_capability": device_capability, "torch_version": torch_version}


def compute_global_rank(tp_size: int, pp_rank: int, tp_rank: int) -> int:

    return tp_size * pp_rank + tp_rank


def compute_local_gpu_id(
    pp_rank: int,
    tp_rank: int,
    pp_size_per_node: int,
    tp_size_per_node: int,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
) -> int:

    return (
        base_gpu_id
        + (pp_rank % pp_size_per_node) * tp_size_per_node
        + (tp_rank % tp_size_per_node) * gpu_id_step
    )


def get_socket_path(global_rank: int) -> str:

    return WEIGHT_CACHE_SOCKET_TEMPLATE.format(global_rank=global_rank)


def get_ready_path(global_rank: int) -> str:

    return WEIGHT_CACHE_READY_TEMPLATE.format(global_rank=global_rank)


def _read_ready_pid(ready_path: str) -> Optional[int]:

    try:
        with open(ready_path) as f:
            for line in f:
                if line.startswith("pid="):
                    return int(line.strip().split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def _is_pid_alive(pid: int) -> bool:

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_stale_daemon_files(global_rank: int, *, force: bool = False) -> None:

    ready_path = get_ready_path(global_rank)
    socket_path = get_socket_path(global_rank)

    if not os.path.exists(ready_path) and not os.path.exists(socket_path):
        return

    pid = _read_ready_pid(ready_path) if os.path.exists(ready_path) else None

    if pid is not None and _is_pid_alive(pid):
        if not force:
            raise RuntimeError(
                f"Weight cache daemon for rank {global_rank} is already running "
                f"(pid={pid}, ready={ready_path}). Stop the existing daemon before "
                f"launching a new one, or pass force=True (--force) to kill it and "
                f"take over."
            )
        logger.warning(
            f"[weight_cache] force takeover: killing existing daemon pid={pid} "
            f"for rank {global_rank} and reclaiming its socket/ready files."
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for path in (ready_path, socket_path):
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Removed stale daemon file: {path}")
