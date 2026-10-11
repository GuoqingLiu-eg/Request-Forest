# RF SGLang

A shared SGLang + Request Forest runtime for **Qwen3.6-35B-A3B**, **Qwen3.8-27B**, and **Ornith-1.5-35B-A3B**, based on commit `4563778e5033dffb81b9aa50e91c7c8397713345`. All three profiles use the same implementation.

## Install

The bundled RF libraries target **Linux x86-64, PyTorch 2.13.0 + CUDA 13, and NVIDIA SM120 GPUs** (the measured system uses RTX 6000D). The CUDA library contains SM120 machine code and no PTX fallback for other GPU architectures. Python 3.12 was used for validation.

From this directory, install the Python runtime into your CUDA 13 environment:

```sh
python -m pip install -e .
```

The RF operators are already compiled:

- `native/request_forest_native_cpu.so`
- `native/request_forest_native_cuda.so`

The launcher loads these files directly. No separate SGLang checkout or RF operator compilation is needed. Model weights and third-party CUDA dependencies are installed separately; the latter are declared in `pyproject.toml`.

## Run

Supply a local target checkpoint and its matching DFlash2 draft checkpoint. Qwen3.6 uses the draft unpacked to BF16.

```sh
python launch.py --profile qwen3.6 --gpus 0,1,2,3 \
  --model-path /path/to/qwen36-target --draft-model-path /path/to/qwen36-draft-bf16

python launch.py --profile qwen3.8 --gpus 0,1,2,3 \
  --model-path /path/to/qwen38-target --draft-model-path /path/to/qwen38-draft

python launch.py --profile ornith --gpus 0,1,2,3 \
  --model-path /path/to/ornith-target --draft-model-path /path/to/ornith-draft
```

Run one profile per GPU group. The default endpoint is `127.0.0.1:31721`; use `--host` and `--port` to change it. Use `--dry-run` to inspect the effective command and environment without loading models. Additional SGLang options can be appended after `--`. `--native-dir` selects another directory containing the two library filenames above.

## Profiles

`configs/common.json` holds the shared serving settings. The three model files specify the model identity and request coalescing interval.

| Profile | Target model | Idle prefill coalescing |
| --- | --- | ---: |
| `qwen3.6` | Qwen3.6-35B-A3B | 4 ms |
| `qwen3.8` | Qwen3.8-27B | 0 ms |
| `ornith` | Ornith-1.5-35B-A3B | 0 ms |

The supplied settings use TP4, BF16 target/draft weights, **BF16 SSM state**, static memory fraction 0.8, draft window 8, RF verification budget 384, FA4 target attention, and FlashInfer draft attention. The short-extend fused path remains disabled.

Adaptive RF enters at B >= 96 active requests and exits at B <= 80. Entry at the high-load boundary is immediate; the normal exit uses two confirmations and a minimum residence of four rounds. A single active request uses the native path. These policy rules are implemented in the shared worker.

Keep `--mamba-ssm-dtype bfloat16` when reproducing the supplied configuration.

### FlashInfer planning optimization

The measured setup also used two CPU reduction changes in FlashInfer 0.6.17. To prepare the same optimization from the installed package:

```sh
python scripts/prepare_flashinfer.py
```

This creates an isolated copy under `build/deps/flashinfer` without modifying the installed package or compiling RF operators. The launcher automatically uses that copy when present.

## Four-benchmark sweep

Run GSM8K, MATH-500, HumanEval, and MT-Bench with static memory fraction **0.8** and client concurrency **64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 240, 256**.

Supply an existing data directory containing `gsm8k.jsonl`, `math500.jsonl`, `humaneval.jsonl`, and `mtbench.jsonl` in their original dataset format, or download the four inputs:

```sh
python scripts/benchmark_data.py --download-dir ./data
```

Launch a model and run all 48 measurements:

```sh
python scripts/run_benchmarks.py \
  --profile qwen3.6 \
  --model-path /path/to/qwen36-target \
  --draft-model-path /path/to/qwen36-draft-bf16 \
  --data-dir ./data \
  --gpus 0,1,2,3 \
  --port 31721
```

Use `--profile qwen3.8` or `--profile ornith` with the corresponding checkpoints for the other models. `--dry-run` prints the plan without starting a server, downloading data, or loading a model.

The server starts once with capacity 256 and all requested CUDA graph batch sizes, including 240. Each measurement runs `4 × B` conversations at client concurrency B. Inputs are shuffled with seed 42 and repeated in that order as needed. Generation is greedy, with EOS enabled, thinking disabled, and a limit of 4096 output tokens per turn. MT-Bench uses both turns and retains the model's first answer as history. Eight short requests warm up the server before the sweep; the cache is flushed before each measurement.

Outputs go to a new `results/<profile>-<timestamp>/` directory, or a new directory specified by `--output-dir`:

- `summary.csv` and `summary.json`: throughput, latency, MAT, and draft acceptance.
- `b*/<suite>.jsonl` and `<suite>.metrics.json`: raw conversations and aggregate metrics.
- `server.log`, `server_info.json`, `protocol.json`, `status.json`: configuration and progress.

Throughput is output tokens divided by queue duration. MAT is `output_tokens / spec_verify_ct`, including the target token. Draft acceptance is accepted / proposed draft tokens, excluding it. Missing speculative counters remain empty. This runner measures serving performance; it does not grade answers or execute generated code.

The script stops only the server process group it starts. Use an unused port and a dedicated GPU group.

## Contents and provenance

- `python/sglang/`: text-serving runtime and RF implementation.
- `native/`: the two existing compiled RF libraries.
- `configs/` and `launch.py`: portable model profiles and server entry point.
- `scripts/run_benchmarks.py`: the four-benchmark sweep and its data/client helpers.
- `PROVENANCE.json`: source revision, profile sources, binary hashes, and validation scope.

The distribution includes the HTTP text-serving path and runtime kernel resources. Historical experiment scripts, standalone tests and microbenchmarks, results, model weights, and native build tools are omitted. The four-benchmark serving runner is included under `scripts/`.

Validation covered source equivalence after comment removal, isolated Python package installation with existing dependencies, the server CLI, registration of eight RF operator interfaces from the bundled libraries, and all three launcher profiles. Model inference was not rerun for this packaging change.

Upstream copyright and license notices are retained. See `LICENSE` and the notices in the corresponding files.
