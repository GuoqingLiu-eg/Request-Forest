import glob
import os
import warnings

from sglang.srt.environ import envs

_CUDA_COREDUMP_FLAGS = (
    "skip_nonrelocated_elf_images,skip_global_memory,"
    "skip_shared_memory,skip_local_memory,skip_constbank_memory"
)


def is_enabled() -> bool:
    return envs.SGLANG_CUDA_COREDUMP.get()


def get_dump_dir() -> str:


    explicit = envs.SGLANG_CUDA_COREDUMP_DIR.get()
    runner_temp = os.getenv("RUNNER_TEMP")
    if explicit:
        base = explicit
    elif runner_temp:
        base = os.path.join(runner_temp, "sglang_cuda_coredumps")
    else:
        base = "/tmp/sglang_cuda_coredumps"


    run_id = os.getenv("GITHUB_RUN_ID")
    if run_id:
        attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
        return os.path.join(base, f"{run_id}-{attempt}")
    return base


def _inject_env():

    dump_dir = get_dump_dir()
    os.makedirs(dump_dir, exist_ok=True)

    env_vars = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": _CUDA_COREDUMP_FLAGS,
        "CUDA_COREDUMP_FILE": f"{dump_dir}/cuda_coredump_%h.%p.%t",
    }
    for key, value in env_vars.items():
        if key in os.environ:
            warnings.warn(
                f"CUDA coredump env var {key} is already set to "
                f"'{os.environ[key]}', skipping injection of '{value}'.",
                stacklevel=2,
            )
        else:
            os.environ[key] = value


def cleanup_dump_dir():

    dump_dir = get_dump_dir()
    for f in glob.glob(os.path.join(dump_dir, "cuda_coredump_*")):
        os.remove(f)


def report():

    dump_dir = get_dump_dir()
    coredump_files = glob.glob(os.path.join(dump_dir, "cuda_coredump_*"))
    if not coredump_files:
        return

    print(f"\n{'='*60}")
    print(f"CUDA coredump(s) detected ({len(coredump_files)} file(s)):")
    for f in coredump_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
    print("Use cuda-gdb to analyze: cuda-gdb -c <coredump_file>")

    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        repo = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang")
        print(f"Download from CI: gh run download {run_id} --repo {repo}")

    print(f"{'='*60}\n")


_SENTINEL = "_SGLANG_CUDA_COREDUMP_INJECTED"

if is_enabled() and _SENTINEL not in os.environ:
    os.environ[_SENTINEL] = "1"
    print(f"Injecting CUDA coredump env vars (pid={os.getpid()})")
    _inject_env()
