"""
POC5 — Response Cache benchmark.

Shows two things:
  A) the same prompt twice  -> miss (slow) vs hit (instant), and the speedup
  B) a realistic workload with repeats -> hit rate + total time saved

Run (with cache_server.py running on port 8000):
    python benchmark.py
"""

import time

import requests

BASE = "http://localhost:8000"


def call(message: str):
    start = time.time()
    data = requests.post(f"{BASE}/chat", json={"message": message}).json()
    return data["cached"], time.time() - start


def main():
    try:
        requests.get(f"{BASE}/", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Server not reachable at {BASE} — is cache_server.py running? ({e})")

    requests.post(f"{BASE}/cache/clear")  # start clean

    print("=== A. Same prompt twice: cache MISS vs cache HIT ===")
    prompt = "Explain KV cache in simple way"
    cached1, t1 = call(prompt)
    cached2, t2 = call(prompt)
    print(f"  1st call: cached={cached1!s:<5} latency={t1:.4f}s   (MISS -> ran the model on the GPU)")
    print(f"  2nd call: cached={cached2!s:<5} latency={t2:.5f}s  (HIT  -> returned from memory)")
    if t2 > 0:
        print(f"  => the cache hit was ~{t1 / t2:,.0f}x faster")

    print("\n=== B. Realistic workload with repeats ===")
    requests.post(f"{BASE}/cache/clear")  # isolate Part B's stats from Part A
    # 3 unique prompts, but they repeat -> a real app sees the same questions a lot.
    workload = [
        "What is batching in LLM inference?",      # miss
        "Why GPU memory matters in inference?",    # miss
        "What is time to first token?",            # miss
        "What is batching in LLM inference?",      # hit
        "Why GPU memory matters in inference?",    # hit
        "What is batching in LLM inference?",      # hit
        "What is time to first token?",            # hit
        "What is batching in LLM inference?",      # hit
    ]

    total = 0.0
    miss_times = []
    for i, m in enumerate(workload, 1):
        cached, t = call(m)
        total += t
        if not cached:
            miss_times.append(t)
        flag = "HIT " if cached else "MISS"
        print(f"  #{i:>2} [{flag}] {t:.5f}s  | {m}")

    avg_miss = sum(miss_times) / len(miss_times) if miss_times else 0
    no_cache_estimate = avg_miss * len(workload)  # if every request had to run the model
    saved = no_cache_estimate - total

    stats = requests.get(f"{BASE}/cache/stats").json()
    print("\nSUMMARY")
    print(f"  requests           : {len(workload)}")
    print(f"  cache stats        : {stats}")
    print(f"  avg miss latency   : {avg_miss:.2f}s  (cost of actually running the model)")
    print(f"  total time WITH cache    : {total:.2f}s")
    print(f"  est. time WITHOUT cache  : {no_cache_estimate:.2f}s  ({len(workload)} model runs)")
    print(f"  => time saved      : ~{saved:.1f}s ({stats['hit_rate_percent']}% of requests were free)")


if __name__ == "__main__":
    main()
