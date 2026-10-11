#!/usr/bin/env python3
import argparse
import importlib.metadata
import importlib.util
import shutil
import tempfile
from pathlib import Path

REPLACEMENTS = (
    ("self._max_q_len = max(qo_indptr_host[1:] - qo_indptr_host[:-1]).item()", "self._max_q_len = (qo_indptr_host[1:] - qo_indptr_host[:-1]).max().item()"),
    ("self._max_kv_len = max(kv_lens_arr_host).item()", "self._max_kv_len = kv_lens_arr_host.max().item()"),
)

def main():
    parser = argparse.ArgumentParser(description="Create an isolated FlashInfer 0.6.17 copy with the measured CPU planning optimizations.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "build" / "deps")
    args = parser.parse_args()
    version = importlib.metadata.version("flashinfer-python")
    if version != "0.6.17":
        parser.error(f"Expected flashinfer-python 0.6.17, found {version}")
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        parser.error("flashinfer-python is not installed")
    source = Path(spec.origin).resolve().parent
    output = args.output_dir.expanduser().resolve()
    destination = output / "flashinfer"
    if destination.exists():
        text = (destination / "prefill.py").read_text()
        if all(new in text and old not in text for old, new in REPLACEMENTS):
            print(destination)
            return
        parser.error(f"Existing destination is not the expected patched package: {destination}")
    text = (source / "prefill.py").read_text()
    for old, new in REPLACEMENTS:
        if text.count(old) == 1 and new not in text:
            text = text.replace(old, new)
        elif old not in text and text.count(new) == 1:
            pass
        else:
            parser.error("Installed FlashInfer planning source does not match the expected version")
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".flashinfer-", dir=output))
    staging = temporary / "flashinfer"
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging / "prefill.py").write_text(text)
    staging.rename(destination)
    temporary.rmdir()
    print(destination)

if __name__ == "__main__":
    main()
