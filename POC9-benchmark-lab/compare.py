"""
POC9 — Compare saved benchmark runs.

Reads every results/*.json produced by bench.py and prints one comparison
table, plus a tiny ASCII bar chart of tokens/sec so differences are obvious.

    python compare.py
"""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        raise SystemExit("No results yet — run bench.py first.")

    runs = [json.load(open(f)) for f in files]

    print(f"\n{'label':<16} {'conc':>4} {'stream':>6} {'req/s':>6} {'tok/s':>7} "
          f"{'p50':>6} {'p95':>6} {'p99':>6} {'TTFT':>6}")
    print("-" * 74)
    for r in runs:
        c, m = r["config"], r["metrics"]
        print(f"{r['label']:<16} {c['concurrency']:>4} {str(c['streaming']):>6} "
              f"{m['req_per_s']:>6} {m['tokens_per_s']:>7} "
              f"{m['p50_s']:>6} {m['p95_s']:>6} {m['p99_s']:>6} {m['ttft_mean_s']:>6}")

    # ASCII bar chart of tokens/sec
    print("\ntokens/sec:")
    peak = max(r["metrics"]["tokens_per_s"] for r in runs) or 1
    for r in runs:
        tps = r["metrics"]["tokens_per_s"]
        bar = "█" * int((tps / peak) * 40)
        print(f"  {r['label']:<16} {bar} {tps}")

    print("\nTTFT (time to first token, lower = snappier):")
    for r in runs:
        ttft = r["metrics"]["ttft_mean_s"]
        bar = "█" * int(min(ttft, 5) / 5 * 40)
        print(f"  {r['label']:<16} {bar} {ttft}s")


if __name__ == "__main__":
    main()
