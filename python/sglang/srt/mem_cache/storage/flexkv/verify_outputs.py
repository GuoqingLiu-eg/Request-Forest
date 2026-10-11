from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPTS = [
    (
        "PROMPT_SHORT",
        "The capital of France is",
        12,
    ),
    (
        "PROMPT_MEDIUM",
        "List the first ten prime numbers in order: 2, 3, 5, ",
        24,
    ),
    (
        "PROMPT_LONG",

        (
            "In the year 2025, a research team at a major AI lab released a "
            "report describing the architecture of a new large language "
            "model. The report had several sections. Section one introduced "
            "the model and its training data. Section two covered the "
            "attention mechanism in detail, including how the keys and "
            "values were managed. Section three discussed deployment, "
            "including KV cache offloading to CPU memory and to disk. "
            "Section four reported evaluation results on standard "
            "benchmarks. Section five concluded with a discussion of "
            "future work, including improvements to the offloading layer "
            "and to the radix tree used to index cached prefixes. "
            "Now, summarize the report in one sentence: "
        ),
        60,
    ),
]


def _post(host: str, path: str, body=None, timeout=120) -> str:
    if body is None:
        req = urllib.request.Request(f"http://{host}{path}", method="POST")
    else:
        req = urllib.request.Request(
            f"http://{host}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def gen(host: str, text: str, max_new: int) -> dict:
    raw = _post(
        host,
        "/generate",
        {
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new,
                "temperature": 0.0,
            },
        },
    )
    return json.loads(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--host",
        default="127.0.0.1:30000",
        help="sglang server host:port (default 127.0.0.1:30000)",
    )
    ap.add_argument(
        "--phase",
        choices=["baseline", "test"],
        required=True,
        help="baseline: record golden outputs; test: compare against them",
    )
    ap.add_argument(
        "--baseline-file",
        default="/tmp/flexkv_baseline.json",
        help="where to write/read the baseline outputs",
    )
    args = ap.parse_args()

    if args.phase == "baseline":
        result = {}
        for name, text, max_new in PROMPTS:
            r = gen(args.host, text, max_new)
            meta = r["meta_info"]
            print(
                f"[baseline] {name}: completion={meta['completion_tokens']}, "
                f"cached={meta['cached_tokens']}, text={r['text']!r}"
            )
            result[name] = {
                "text": r["text"],
                "output_ids": r["output_ids"],
                "completion_tokens": meta["completion_tokens"],
            }
        with open(args.baseline_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote baseline to {args.baseline_file}")
        return 0

    with open(args.baseline_file) as f:
        baseline = json.load(f)

    errors = 0
    for name, text, max_new in PROMPTS:
        b = baseline[name]


        r1 = gen(args.host, text, max_new)
        m1 = r1["meta_info"]
        ok1 = r1["output_ids"] == b["output_ids"]
        print(
            f"[test/{name}] R1 fresh: cached={m1['cached_tokens']}/"
            f"{m1['prompt_tokens']}, details={m1.get('cached_tokens_details')}, "
            f"output_match={'OK' if ok1 else 'MISMATCH'}"
        )
        if not ok1:
            print(f"  baseline: {b['text']!r}")
            print(f"  r1      : {r1['text']!r}")
            errors += 1


        time.sleep(2)
        _post(args.host, "/flush_cache")
        time.sleep(1)


        r2 = gen(args.host, text, max_new)
        m2 = r2["meta_info"]
        ok2 = r2["output_ids"] == b["output_ids"]
        ratio = m2["cached_tokens"] / max(1, m2["prompt_tokens"])
        print(
            f"[test/{name}] R2 cached: cached={m2['cached_tokens']}/"
            f"{m2['prompt_tokens']} ({ratio:.1%}), "
            f"details={m2.get('cached_tokens_details')}, "
            f"output_match={'OK' if ok2 else 'MISMATCH'}"
        )
        if not ok2:
            print(f"  baseline: {b['text']!r}")
            print(f"  r2      : {r2['text']!r}")
            errors += 1

    print(f"\nTotal mismatches: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
