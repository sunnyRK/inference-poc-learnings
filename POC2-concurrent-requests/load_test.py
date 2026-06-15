"""
POC2 — Concurrent Request Benchmark

Fires N requests at the POC1 inference server at several concurrency
levels and measures what happens to throughput and latency under load.

We use a thread pool (not asyncio) on purpose: `requests` is blocking, but
it releases the GIL while waiting on the network, so threads give us real
*in-flight* concurrency against the server with zero extra dependencies.

A ThreadPoolExecutor with `max_workers=C` keeps exactly C requests in
flight at once (the rest queue on the client), which is precisely how we
cap server-side concurrency at C.

Run:  python load_test.py
      python load_test.py --levels 1,2,4,8,16 --requests 12
"""

import argparse
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL = "http://localhost:8000/chat"

# Reuse the POC1 prompt set; we cycle through it to build the workload.
PROMPTS = [
    "Explain KV cache in simple way",
    "What is batching in LLM inference?",
    "Explain tokenization in 2 lines",
    "What is time to first token?",
    "Why GPU memory matters in inference?",
]


def one_request(prompt: str) -> dict:
    """Send a single /chat request and time it end-to-end (client side)."""
    start = time.time()
    try:
        res = requests.post(URL, json={"message": prompt}, timeout=180)
        data = res.json()
        return {
            "ok": True,
            "latency": time.time() - start,
            "tokens": data.get("eval_count") or 0,
        }
    except Exception as e:  # noqa: BLE001 - we want to record any failure
        return {"ok": False, "latency": time.time() - start, "tokens": 0, "error": str(e)}


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100). p95 = the value 95% of requests beat."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_level(concurrency: int, total_requests: int) -> dict:
    """Run `total_requests` through a pool of size `concurrency`; collect stats."""
    workload = [PROMPTS[i % len(PROMPTS)] for i in range(total_requests)]

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, p) for p in workload]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.time() - wall_start

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)

    return {
        "concurrency": concurrency,
        "requests": total_requests,
        "ok": len(ok),
        "failed": total_requests - len(ok),
        "wall": wall,
        "req_per_sec": len(ok) / wall if wall else 0,
        "tokens_per_sec": total_tokens / wall if wall else 0,
        "mean": statistics.fmean(latencies) if latencies else 0,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "min": min(latencies) if latencies else 0,
        "max": max(latencies) if latencies else 0,
    }


def warmup():
    """One throwaway request so the cold-start model-load tax doesn't pollute level 1."""
    print("Warming up (loading model into memory)...")
    one_request(PROMPTS[0])


def main():
    parser = argparse.ArgumentParser(description="Concurrent inference load test")
    parser.add_argument("--levels", default="1,2,4,8",
                        help="comma-separated concurrency levels, e.g. 1,2,4,8,16")
    parser.add_argument("--requests", type=int, default=8,
                        help="total requests sent at each level")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    # Sanity check the server is reachable before doing real work.
    try:
        requests.get("http://localhost:8000/", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Server not reachable at {URL} — is POC1 running? ({e})")

    warmup()

    rows = []
    for c in levels:
        print(f"\n=== concurrency = {c}  ({args.requests} requests) ===")
        stats = run_level(c, args.requests)
        rows.append(stats)
        print(f"  wall time     : {stats['wall']:.2f} s")
        print(f"  throughput    : {stats['req_per_sec']:.2f} req/s   |   "
              f"{stats['tokens_per_sec']:.1f} tokens/s")
        print(f"  latency  mean : {stats['mean']:.2f} s")
        print(f"  latency  p50  : {stats['p50']:.2f} s   "
              f"p95: {stats['p95']:.2f} s   p99: {stats['p99']:.2f} s")
        if stats["failed"]:
            print(f"  FAILED        : {stats['failed']} request(s)")

    # Summary table — the degradation curve at a glance.
    print("\n" + "=" * 78)
    print("SUMMARY — throughput & latency vs concurrency")
    print("=" * 78)
    print(f"{'conc':>4} | {'req/s':>7} | {'tok/s':>7} | {'p50 (s)':>8} | "
          f"{'p95 (s)':>8} | {'p99 (s)':>8} | {'failed':>6}")
    print("-" * 78)
    base_tps = rows[0]["tokens_per_sec"] or 1
    for r in rows:
        print(f"{r['concurrency']:>4} | {r['req_per_sec']:>7.2f} | "
              f"{r['tokens_per_sec']:>7.1f} | {r['p50']:>8.2f} | "
              f"{r['p95']:>8.2f} | {r['p99']:>8.2f} | {r['failed']:>6}")
    print("-" * 78)
    peak = max(rows, key=lambda r: r["tokens_per_sec"])
    print(f"Peak throughput: {peak['tokens_per_sec']:.1f} tok/s at concurrency {peak['concurrency']} "
          f"({peak['tokens_per_sec'] / base_tps:.2f}x the single-stream baseline)")


if __name__ == "__main__":
    main()
