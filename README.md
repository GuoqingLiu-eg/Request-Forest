# Request Forest

A SGLang-based Request Forest speculative decoding runtime for
**Qwen3.6-35B-A3B**, **Qwen3.8-27B**, and **Ornith-1.5-35B-A3B**.
The three profiles share the same text-serving implementation and load-aware
execution policy.

This repository contains the archived serving package, RF CPU/CUDA operator
sources, build entry points, model profiles, and upstream license notices.
Model weights, datasets, compiled libraries, and historical experiment results
are stored separately.

## Installation

The recorded deployment used Linux x86-64, Python 3.12, PyTorch 2.13.0 with
CUDA 13, and NVIDIA SM120 GPUs (RTX 6000D). Dependencies are declared in
`pyproject.toml`. Prepare a matching CUDA-enabled PyTorch environment and a
C++17/NVCC/Ninja toolchain before installation.

```sh
git clone https://github.com/GuoqingLiu-eg/Request-Forest.git
cd Request-Forest
python -m pip install -e .
```

Build the two RF libraries from the supplied sources:

```sh
python scripts/request_forest/load_native_cpu_ops.py \
  --build-directory build/native
python scripts/request_forest/load_native_cuda_ops.py \
  --build-directory build/native
```

The build directory will contain `request_forest_native_cpu.so` and
`request_forest_native_cuda.so`. These libraries must be compiled for the
target machine; this source release does not include precompiled binaries.
The CPU builder also supports `--include-eagle`, which produces a differently
named library for that optional build variant.

## Launch a model

Provide a local Target checkpoint and its matching DFlash2 Draft checkpoint.
For the recorded Qwen3.6 deployment, the Draft weights were unpacked to BF16.

Inspect the configuration without loading models:

```sh
python launch.py --profile qwen3.6 --gpus 0,1,2,3 \
  --model-path /path/to/target --draft-model-path /path/to/draft \
  --native-dir build/native --dry-run
```

Remove `--dry-run` to start the server. The other supported profiles are
`qwen3.8` and `ornith`. The default endpoint is `127.0.0.1:31721`; use
`--host` and `--port` to change it. Additional SGLang options may be appended
after `--`. Choose an available GPU group and port for each server.

## Supplied configuration

| Setting | Default |
| --- | --- |
| Tensor parallelism | 4 GPUs |
| Target / Draft weights | BF16 |
| SSM state | BF16 |
| Static memory fraction | 0.8 |
| Draft window | 8 |
| RF verification budget | 384 |
| Target attention | FA4 |
| Draft attention | FlashInfer |

Adaptive RF enters at 96 active requests and exits at 80. Entry at the
high-load boundary is immediate; normal exit uses two confirmations and a
minimum residence of four rounds. A single active request uses the native
path. The short-extend fused path remains disabled.

`configs/common.json` holds the shared settings. Model-specific settings are
in `configs/qwen3.6.json`, `configs/qwen3.8.json`, and `configs/ornith.json`.
The Qwen3.6 profile uses a 4 ms idle prefill coalescing interval; the other
two profiles use 0 ms. Preserve `--mamba-ssm-dtype bfloat16` when reproducing
these profiles.

## Optional FlashInfer preparation

The archived deployment used an isolated planning optimization for
FlashInfer 0.6.17:

```sh
python scripts/prepare_flashinfer.py
```

The helper creates a copy under `build/deps/flashinfer`. The launcher uses it
when present. See the helper and the FlashInfer notices for its scope.

## Repository layout

| Path | Contents |
| --- | --- |
| `python/sglang/` | Serving runtime, RF workers, and runtime kernel resources |
| `python/sglang/srt/speculative/request_forest/` | RF scheduling, tree handling, native loading, and worker implementation |
| `python/sglang/kernels/aot/csrc/` | RF CPU/CUDA sources and optional CPU tree builder |
| `benchmark/request_forest/` | Native operator registration sources |
| `scripts/request_forest/` | CPU/CUDA build entry points |
| `configs/`, `launch.py` | Shared launcher and three model profiles |
| `licenses/` | Third-party notices and license copies |
| `docs/` | Historical package documentation and provenance |

## Provenance and validation

The serving package comes from Request Forest commit
`4563778e5033dffb81b9aa50e91c7c8397713345`, based on SGLang commit
`f8cc1f9525c3a0bf3b14480cc76eccb79db1b4ea`.
See [PROVENANCE.json](PROVENANCE.json) for the release scope.

The historical package README and provenance are preserved under `docs/`.
Their precompiled-library and benchmark-runner descriptions refer to the
original distribution; those artifacts were excluded from the subsequent
source archive and are not included here. Use this README for the current
source build procedure.

This export receives static source and launcher checks. Native compilation,
dependency installation, and model inference have not been rerun from this
GitHub checkout. Recorded environment settings are not a compatibility claim
for other GPUs or CUDA/PyTorch combinations.

Upstream copyright and license notices are retained. See [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the notices in individual
source files.
