import importlib
import json
import logging
import os
import random
import ssl
import subprocess
import sys
import time
import traceback
import urllib.request
import warnings
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, wraps
from io import BytesIO
from json import dumps
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pybase64
import requests
from IPython.display import HTML, display
from pydantic import BaseModel
from tqdm import tqdm

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def load_diffusion_overlay_registry_from_env() -> dict[str, dict[str, Any]]:
    raw_value = os.getenv("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY", "").strip()
    if not raw_value:
        return {}

    if raw_value.startswith("{"):
        payload = json.loads(raw_value)
    else:
        with open(os.path.expanduser(raw_value), encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for source_model_id, spec in payload.items():
        if isinstance(spec, str):
            normalized[source_model_id] = {"overlay_repo_id": spec}
        elif isinstance(spec, dict) and spec.get("overlay_repo_id"):
            normalized[source_model_id] = dict(spec)
    return normalized


def has_diffusion_overlay_registry_match(
    model_path: str, registry: dict[str, dict[str, Any]] | None = None
) -> bool:
    registry = (
        load_diffusion_overlay_registry_from_env() if registry is None else registry
    )
    if model_path in registry:
        return True
    if not os.path.exists(model_path):
        return False
    base_name = os.path.basename(os.path.normpath(model_path))
    return any(base_name == key.rsplit("/", 1)[-1] for key in registry)


def execute_once(func):
    has_run = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            func(*args, **kwargs)
            has_run = True

    return wrapper


@execute_once
def info_once(message: str):
    logger.info(message)


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:

    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def is_same_type(values: list):

    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


def read_jsonl(filename: str):

    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def dump_state_text(filename: str, states: list, mode: str = "w"):

    from sglang.lang.interpreter import ProgramState

    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                pass
            elif isinstance(s, ProgramState):
                s = s.text()
            else:
                s = str(s)

            fout.write(
                "=" * 40 + f" {i} " + "=" * 40 + "\n" + s + "\n" + "=" * 80 + "\n\n"
            )


def normalize_base_url(host: str, port: int) -> str:
    from sglang.srt.utils.network import NetworkAddress

    if host.startswith("http://") or host.startswith("https://"):
        warnings.warn(
            f"Including the scheme in --host ('{host}') is deprecated. "
            f"Pass just the hostname (e.g. '127.0.0.1') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return f"{host}:{port}"
    return NetworkAddress(host, port).to_url()


class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    @cached_property
    def _body(self):
        return self.resp.read()

    def json(self):
        return json.loads(self._body)

    @property
    def text(self):
        return self._body.decode("utf-8", errors="replace")

    @property
    def status_code(self):
        return self.resp.status


def http_request(
    url,
    json=None,
    stream=False,
    api_key=None,
    verify=None,
    method: Optional[str] = None,
):

    headers = {"Content-Type": "application/json; charset=utf-8"}


    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
        if json is None:
            data = None
        else:
            data = bytes(dumps(json), encoding="utf-8")

        try:
            if sys.version_info >= (3, 13):

                if verify and isinstance(verify, str):
                    context = ssl.create_default_context(cafile=verify)
                else:
                    context = ssl.create_default_context()
                resp = urllib.request.urlopen(req, data=data, context=context)
            else:
                resp = urllib.request.urlopen(req, data=data, cafile=verify)
            return HttpResponse(resp)
        except urllib.error.HTTPError as e:
            return HttpResponse(e)


def encode_image_base64(image_path: Union[str, bytes]):

    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return pybase64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return pybase64.b64encode(image_path).decode("utf-8")
    else:
        import torch

        if isinstance(image_path, torch.Tensor):

            from PIL import Image

            tensor = image_path.cpu() if image_path.device.type != "cpu" else image_path
            image_path = Image.fromarray(tensor.permute(1, 2, 0).numpy())


        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return pybase64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_frame(frame):
    import cv2
    from PIL import Image


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    im_pil = Image.fromarray(frame)


    buffered = BytesIO()


    im_pil.save(buffered, format="PNG")

    frame_bytes = buffered.getvalue()


    return frame_bytes


def encode_video_base64(video_path: str, num_frames: int = 16):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file:{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"target_frames: {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:


            pass

    cap.release()


    frames = [frames[i] for i in frame_indices if i < len(frames)]


    while len(frames) < num_frames:
        frames.append(frames[-1])


    with ThreadPoolExecutor() as executor:
        encoded_frames = list(executor.map(encode_frame, frames))


    video_bytes = b"".join(encoded_frames)


    video_base64 = "video:" + pybase64.b64encode(video_bytes).decode("utf-8")

    return video_base64


def _is_chinese_char(cp: int):


    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True

    return False


def find_printable_text(text: str):


    if text.endswith("\n"):
        return text

    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text

    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]


    else:
        return text[: text.rfind(" ") + 1]


class LazyImport:


    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


def download_and_cache_file(url: str, filename: Optional[str] = None):

    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])


    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")


    response = requests.get(url, stream=True)
    response.raise_for_status()


    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024


    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_in_ci() -> bool:
    return envs.SGLANG_IS_IN_CI.get()


def print_highlight(html_content: str):
    if is_in_ci():
        html_content = str(html_content).replace("\n", "<br>")
        display(HTML(f"<strong style='color: #00008B;'>{html_content}</strong>"))
    else:
        print(html_content)


process_socket_map = weakref.WeakKeyDictionary()


def reserve_port(host, start=30000, end=40000):

    from sglang.srt.utils.network import try_bind_socket

    candidates = list(range(start, end))
    random.shuffle(candidates)
    for port in candidates:
        try:
            sock = try_bind_socket(host, port)
            return port, sock
        except OSError:
            continue
    raise RuntimeError("No free port available.")


def release_port(lock_socket):

    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")


def execute_shell_command(command: str) -> subprocess.Popen:

    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" in part and not part.startswith("-") and not part.startswith("/"):
            key, _, value = part.partition("=")
            if key and value is not None and key.replace("_", "").isalnum():
                env[key] = value
                i += 1
                continue
        break
    parts = parts[i:]
    if not parts:
        raise ValueError(
            "Command contains only environment variable assignments, no executable"
        )
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):

    if port is None:
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)

    if lock_socket is not None:
        process_socket_map[process] = lock_socket

    return process, port


def terminate_process(process):

    from sglang.srt.utils import kill_process_tree

    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)


def _raise_if_process_exited(process: Optional[Any]) -> None:
    if process is None:
        return

    if hasattr(process, "poll"):
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited with code {return_code}")
        return

    if hasattr(process, "is_alive") and not process.is_alive():
        return_code = getattr(process, "exitcode", None)
        if return_code is None:
            raise RuntimeError("Server process exited")
        raise RuntimeError(f"Server process exited with code {return_code}")


def _is_wait_timeout(start_time: float, timeout: Optional[int]) -> bool:
    if timeout is None:
        return False
    return time.perf_counter() - start_time > timeout


def wait_for_http_ready(
    url: str,
    timeout: Optional[int] = None,
    process: Optional[Any] = None,
    headers: Optional[dict] = None,
    request_timeout: int = 5,
) -> None:

    start_time = time.perf_counter()
    while True:
        _raise_if_process_exited(process)
        try:
            response = requests.get(url, headers=headers, timeout=request_timeout)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            _raise_if_process_exited(process)

        if _is_wait_timeout(start_time, timeout):
            raise TimeoutError(
                f"Endpoint {url} did not become ready within timeout period"
            )
        time.sleep(1)


def wait_for_server(
    base_url: str,
    timeout: int = None,
    process: Optional[subprocess.Popen] = None,
) -> None:

    wait_for_http_ready(
        url=f"{base_url}/v1/models",
        timeout=timeout,
        process=process,
        headers={"Authorization": "Bearer None"},
    )
    time.sleep(5)
    print_highlight("""\n
        NOTE: Typically, the server runs in a separate terminal.
        In this notebook, we run the server and notebook code together, so their outputs are combined.
        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
        To reduce the log length, we set the log level to warning for the server, the default log level is info.
        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.
        """)


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):


        self._mapping = OrderedDict(mapping)

        self._mro_cache = {}
        self._fallback_fn = None

    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    def __iadd__(self, other: "TypeBasedDispatcher"):
        for ty, fn in other._mapping.items():
            if ty not in self._mapping:
                self._mapping[ty] = fn

        self._mro_cache.clear()
        return self

    def __call__(self, obj: Any):
        obj_type = type(obj)

        fn = self._mapping.get(obj_type)
        if fn is not None:
            return fn(obj)


        cached_fn = self._mro_cache.get(obj_type)
        if cached_fn is not None:
            return cached_fn(obj)


        for ty, fn in self._mapping.items():
            if isinstance(obj, ty):
                self._mro_cache[obj_type] = fn
                return fn(obj)


        self._mro_cache[obj_type] = None

        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")


def trim_overlap(existing_text, new_chunk):

    max_overlap = 0
    max_possible = min(len(existing_text), len(new_chunk))
    for i in range(max_possible, 0, -1):
        if existing_text.endswith(new_chunk[:i]):
            max_overlap = i
            break
    return new_chunk[max_overlap:]


def stream_and_merge(llm, prompt, sampling_params):

    final_text = ""
    for chunk in llm.generate(prompt, sampling_params, stream=True):
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
    return final_text


async def async_stream_and_merge(llm, prompt, sampling_params):

    final_text = ""
    generator = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in generator:
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
        yield cleaned_chunk


def resolve_obj_by_qualname(qualname: str) -> Any:

    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
