# SPDX-License-Identifier: Apache-2.0


import logging
import os
import signal
import stat
import threading
import time
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
)

from .protocol import (
    CacheConfig,
    check_ipc_quant_support,
    compute_env_stamp,
    get_quant_method_name,
    hash_quant_config,
    recv_msg,
    send_msg,
)
from .transport import TORCH_IPC_BACKEND, get_client_transport_backend

logger = logging.getLogger(__name__)


_DAEMON_LIVENESS_POLL_INTERVAL = 5.0


class IpcModelLoader(BaseModelLoader):


    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: str,
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str = "auto",
    ):
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format
        self._transport_backend = get_client_transport_backend(TORCH_IPC_BACKEND)

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> nn.Module:

        tic = time.perf_counter()


        quant_method, engine_quant_config = self._resolve_engine_quant(model_config)
        check_ipc_quant_support(quant_method, engine_quant_config, where="client")


        cache_data = self._fetch_from_cache(model_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                raise RuntimeError(
                    f"[IpcModelLoader] Weight cache daemon not available at "
                    f"{self.socket_path}. In daemon mode, fallback to disk "
                    f"loading is disabled because the daemon process already "
                    f"holds weights on the same GPU — loading from disk would "
                    f"cause OOM. Please ensure the weight cache daemon is "
                    f"running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] Weight cache not available or config mismatch, "
                "falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

        entries = cache_data["entries"]
        logger.info(
            f"[IpcModelLoader] Fetched {len(entries)} tensors from daemon "
            f"(transport={self._transport_backend.name}) "
            f"in {time.perf_counter() - tic:.2f}s"
        )

        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        model = self._load_zero_copy_mode(
            model_config,
            device_config,
            entries,
            quant_config,
        )


        self._rebuild_stale_views(model)


        self._start_daemon_liveness_watchdog(cache_data.get("pid"))

        logger.info(
            f"[IpcModelLoader] Loaded model via IPC (mode={self.weight_cache_mode}), "
            f"total={time.perf_counter() - tic:.2f}s"
        )

        return model.eval()

    def _start_daemon_liveness_watchdog(self, daemon_pid: Optional[int]) -> None:

        if not daemon_pid or daemon_pid <= 0:
            logger.warning(
                "[IpcModelLoader] Daemon did not report a PID; skipping the "
                "daemon-liveness watchdog. A daemon crash will not be detected."
            )
            return

        def _daemon_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not _daemon_alive(daemon_pid):
                    logger.critical(
                        f"[IpcModelLoader] Weight cache daemon (pid={daemon_pid}) "
                        f"died while this engine holds its weights via CUDA IPC. "
                        f"The mapped weight pointers are now dangling; continuing "
                        f"would read freed GPU memory. Terminating this process."
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            f"[IpcModelLoader] Started daemon-liveness watchdog for pid={daemon_pid}"
        )

    def _resolve_engine_quant(self, model_config):

        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(model_config.quantization)
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)
        return quant_method, quant_config

    @staticmethod
    def _rebuild_stale_views(model):

        try:
            from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        except ImportError:
            return

        count = 0
        for _, module in model.named_modules():
            conv1d = getattr(module, "conv1d", None)
            attn = getattr(module, "attn", None)
            if conv1d is not None and isinstance(attn, RadixLinearAttention):
                if hasattr(conv1d, "weight") and conv1d.weight is not None:
                    attn.conv_weights = conv1d.weight.view(
                        conv1d.weight.size(0), conv1d.weight.size(2)
                    )
                    if hasattr(conv1d, "bias") and conv1d.bias is not None:
                        attn.bias = conv1d.bias
                    count += 1

        if count > 0:
            logger.info(f"[IpcModelLoader] Rebuilt {count} stale conv_weights views")

    @staticmethod
    def _set_module_tensor(model, name, tensor, is_param=True):

        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf_name = parts[-1]
        if is_param:


            new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(obj, leaf_name, new_param)
        else:


            if leaf_name in obj._parameters:
                del obj._parameters[leaf_name]
            elif hasattr(obj, leaf_name) and leaf_name not in obj._buffers:
                delattr(obj, leaf_name)
            obj.register_buffer(leaf_name, tensor)

    def _load_zero_copy_mode(
        self,
        model_config,
        device_config,
        entries,
        quant_config,
    ) -> nn.Module:

        from sglang.srt.model_loader.utils import set_default_torch_dtype


        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )


        existing_params = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        mismatched = []
        new_params_count = 0
        map_tic = time.perf_counter()


        for name, entry in entries.items():
            imported_tensor = self._transport_backend.import_tensor(entry)
            is_param = entry.get("is_param", True)

            if name in existing_names:

                if name in existing_params:
                    ref_param = existing_params[name]
                else:
                    ref_param = existing_buffers[name]
                if (
                    imported_tensor.shape != ref_param.shape
                    or imported_tensor.dtype != ref_param.dtype
                ):
                    mismatched.append(
                        f"  {name}: IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    continue


            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        if mismatched:
            raise RuntimeError(
                f"[IpcModelLoader] {len(mismatched)} tensor(s) have shape/dtype "
                f"mismatch between the IPC daemon and the meta-initialized model. "
                f"The quantization method passed the IPC allowlist gate "
                f"(check_ipc_quant_support), so this is NOT an unsupported-quant "
                f"case — it indicates the daemon's weight fingerprint is "
                f"incomplete or the daemon/client configs drifted (a bug to fix), "
                f"not merely uninitialized weights:\n" + "\n".join(mismatched)
            )


        still_on_meta_params = [
            name
            for name, param in model.named_parameters()
            if param.device.type == "meta"
        ]
        still_on_meta_buffers = [
            name for name, buf in model.named_buffers() if buf.device.type == "meta"
        ]

        if still_on_meta_params or still_on_meta_buffers:
            raise RuntimeError(
                f"[IpcModelLoader] After IPC mapping, "
                f"{len(still_on_meta_params)} parameter(s) and "
                f"{len(still_on_meta_buffers)} buffer(s) remain on the meta device "
                f"— the daemon did not export them. Refusing to fill them with "
                f"uninitialized memory, which would silently produce wrong output. "
                f"This means the daemon's export is incomplete, or a recomputable "
                f"buffer needs explicit recompute logic here.\n"
                f"  params: {still_on_meta_params[:10]}"
                f"{'...' if len(still_on_meta_params) > 10 else ''}\n"
                f"  buffers: {still_on_meta_buffers[:10]}"
                f"{'...' if len(still_on_meta_buffers) > 10 else ''}"
            )

        map_elapsed = time.perf_counter() - map_tic


        if imported_refs:
            model._ipc_imported_tensors = imported_refs


        model._weight_cache_transport_backend = self._transport_backend

        logger.info(
            f"[IpcModelLoader] Zero-copy: mapped {imported_count} tensors "
            f"({new_params_count} new post-quant), time={map_elapsed:.3f}s"
        )

        return model

    def _fetch_from_cache(self, model_config) -> Optional[dict]:

        import socket as socket_mod


        try:
            st = os.lstat(self.socket_path)
        except FileNotFoundError:
            logger.info(
                f"[IpcModelLoader] Daemon socket not found at {self.socket_path}."
            )
            return None
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(
                f"[IpcModelLoader] Refusing to connect: {self.socket_path} is not "
                f"a socket owned by this user."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)
            sock.connect(self.socket_path)
        except FileNotFoundError:

            sock.close()
            return None
        except ConnectionRefusedError:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Daemon socket exists at {self.socket_path} but "
                f"refused the connection. The daemon may have crashed after "
                f"creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Failed to connect to daemon at "
                f"{self.socket_path}: {e}"
            ) from e

        try:

            from sglang.srt.runtime_context import get_exec, get_parallel

            ps = get_parallel()
            tp_size = ps.tp_size
            tp_rank = ps.tp_rank

            pp_size = ps.pp_size
            pp_rank = ps.pp_rank

            ep_size = ps.moe_ep_size
            moe_dp_size = ps.moe_dp_size
            moe_dp_rank = ps.moe_dp_rank
            moe_ep_rank = ps.moe_ep_rank

            dp_size = get_parallel().config.dp_size

            quant_method, quant_config = self._resolve_engine_quant(model_config)

            engine_config = CacheConfig(
                model_path=model_config.model_path,
                model_arch=(
                    model_config.hf_config.architectures[0]
                    if model_config.hf_config.architectures
                    else ""
                ),
                tp_size=tp_size,
                tp_rank=tp_rank,
                pp_size=pp_size,
                pp_rank=pp_rank,
                dp_size=dp_size,
                ep_size=ep_size,
                moe_dp_size=moe_dp_size,
                moe_dp_rank=moe_dp_rank,
                moe_ep_rank=moe_ep_rank,
                enable_dp_attention=ps.config.enable_dp_attention,
                enable_dp_lm_head=ps.config.enable_dp_lm_head,
                attn_cp_size=ps.attn_cp_size,
                moe_dense_tp_size=ps.config.moe_dense_tp_size,
                moe_a2a_backend=get_exec().moe.moe_a2a_backend,
                quant_method=quant_method,
                quant_config_hash=hash_quant_config(quant_config),
                dtype=str(model_config.dtype),
                revision=model_config.revision or "",
                **compute_env_stamp(),
            )

            logger.info(
                f"[IpcModelLoader] Requesting weights from daemon at "
                f"{self.socket_path} with config: "
                f"model={engine_config.model_path}, "
                f"arch={engine_config.model_arch}, "
                f"tp={engine_config.tp_size}/{engine_config.tp_rank}, "
                f"quant={engine_config.quant_method}, "
                f"dtype={engine_config.dtype}"
            )

            send_msg(sock, {"type": "fetch_state", "config": engine_config.to_dict()})
            result = recv_msg(sock)

            if result.get("status") != "ok":
                daemon_config = result.get("daemon_config", {})
                raise RuntimeError(
                    f"[IpcModelLoader] Daemon config mismatch!\n"
                    f"  Engine config: {engine_config.to_dict()}\n"
                    f"  Daemon config: {daemon_config}"
                )

            backend_name = result.get("transport_backend", TORCH_IPC_BACKEND)
            self._transport_backend = get_client_transport_backend(backend_name)
            result = self._transport_backend.recv_fetch_state_response(sock, result)
            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[IpcModelLoader] Error communicating with daemon at "
                f"{self.socket_path}: {e}"
            ) from e
        finally:
            sock.close()

    def _fallback_load(self, model_config, device_config) -> nn.Module:

        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.loader import DefaultModelLoader

        fallback_config = LoadConfig(
            load_format=self._fallback_load_format,
            download_dir=self.load_config.download_dir,
            model_loader_extra_config=self.load_config.model_loader_extra_config,
            tp_rank=self.load_config.tp_rank,
        )
        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(fallback_config)
        return fallback.load_model(
            model_config=model_config, device_config=device_config
        )

    def download_model(self, model_config) -> None:

        pass
