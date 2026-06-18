"""
POC9 — Benchmark Lab

One reusable load-test tool for any LLM endpoint. It fires N requests at a
chosen concurrency, supports streaming or non-streaming, and reports the
standard serving metrics — then saves the run to results/<label>.json so you
can compare setups later (use compare.py).

Metrics it captures:
  throughput : requests/sec and tokens/sec
  latency    : p50 / p95 / p99 (end-to-end per request)
  TTFT       : time to first token (only meaningful in streaming mode)

It talks to Ollama's /api/chat directly so it works without any of our POC
servers running. Same threaded load generator as POC2.

Examples:
  python bench.py --label noStream-c1 --concurrency 1 --requests 8
  python bench.py --label stream-c4  --concurrency 4 --requests 8 --streaming
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PROMPTS = [
    "Explain KV cache in simple way",
    "What is batching in LLM inference?",
    "Explain tokenization in 2 lines",
    "What is time to first token?",
    "Why GPU memory matters in inference?",
]


def percentile(values, p):
    """Linear-interpolation percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def one_request(prompt, max_tokens, streaming):
    start = time.time()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": streaming,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    try:
        if streaming:
            ttft, tokens = None, 0
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content and ttft is None:
                        ttft = time.time() - start
                    if content:
                        tokens += 1
                    if data.get("done"):
                        tokens = data.get("eval_count") or tokens
                        break
            latency = time.time() - start
            return {"ok": True, "latency": latency, "ttft": ttft or latency, "tokens": tokens}
        else:
            data = requests.post(OLLAMA_URL, json=payload, timeout=180).json()
            latency = time.time() - start
            return {"ok": True, "latency": latency, "ttft": latency,
                    "tokens": data.get("eval_count") or 0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency": time.time() - start, "ttft": 0, "tokens": 0, "error": str(e)}


def run(label, concurrency, n_requests, max_tokens, streaming):
    workload = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]

    one_request(PROMPTS[0], 3, streaming)  # warmup (skip cold start)

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, p, max_tokens, streaming) for p in workload]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.time() - wall_start

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)

    return {
        "label": label,
        "config": {
            "model": MODEL, "concurrency": concurrency, "requests": n_requests,
            "max_tokens": max_tokens, "streaming": streaming,
        },
        "metrics": {
            "ok": len(ok), "failed": n_requests - len(ok), "wall_s": round(wall, 3),
            "req_per_s": round(len(ok) / wall, 2) if wall else 0,
            "tokens_per_s": round(total_tokens / wall, 1) if wall else 0,
            "p50_s": round(percentile(latencies, 50), 3),
            "p95_s": round(percentile(latencies, 95), 3),
            "p99_s": round(percentile(latencies, 99), 3),
            "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0,
        },
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser(description="LLM serving benchmark")
    ap.add_argument("--label", required=True, help="name for this run (saved to results/<label>.json)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Ollama not reachable on :11434 — is it running? ({e})")

    print(f"running '{args.label}': concurrency={args.concurrency} requests={args.requests} "
          f"max_tokens={args.max_tokens} streaming={args.streaming}")
    res = run(args.label, args.concurrency, args.requests, args.max_tokens, args.streaming)

    m = res["metrics"]
    print(f"  throughput : {m['req_per_s']} req/s | {m['tokens_per_s']} tok/s")
    print(f"  latency    : p50 {m['p50_s']}s  p95 {m['p95_s']}s  p99 {m['p99_s']}s")
    print(f"  TTFT mean  : {m['ttft_mean_s']}s")
    if m["failed"]:
        print(f"  FAILED     : {m['failed']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  saved -> results/{args.label}.json")


if __name__ == "__main__":
    main()
