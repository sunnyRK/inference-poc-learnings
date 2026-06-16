"""
POC3 — Streaming vs Non-Streaming benchmark.

The big number we care about here is TTFT = Time To First Token:
"how long until the user sees the FIRST word."

  - Non-streaming: you see nothing until the WHOLE answer is done,
                   so TTFT == total time (you wait the full thing).
  - Streaming:     you see the first word almost immediately,
                   so TTFT is tiny, even though total time is similar.

Run (with stream_server.py running on port 8000):
    python benchmark.py
"""

import json
import time

import requests

BASE = "http://localhost:8000"

PROMPTS = [
    "Explain KV cache in simple way",
    "What is batching in LLM inference?",
    "Why GPU memory matters in inference?",
]


def run_non_streaming(message: str) -> dict:
    """No tokens until the end -> first word and last word arrive at the same moment."""
    start = time.time()
    res = requests.post(f"{BASE}/chat", json={"message": message})
    data = res.json()
    total = time.time() - start
    return {"ttft": total, "total": total, "tokens": data.get("eval_count") or 0}


def run_streaming(message: str, show_live: bool = False) -> dict:
    """Measure when the FIRST token arrives vs when the LAST token arrives."""
    start = time.time()
    ttft = None
    tokens = 0
    with requests.post(f"{BASE}/chat/stream", json={"message": message}, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if data.get("done"):
                continue
            token = data.get("token", "")
            if ttft is None:                  # this is the very first token
                ttft = time.time() - start
            tokens += 1
            if show_live:
                print(token, end="", flush=True)
    total = time.time() - start
    return {"ttft": ttft or total, "total": total, "tokens": tokens}


def main():
    # Make sure the server is up first.
    try:
        requests.get(f"{BASE}/", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Server not reachable at {BASE} — is stream_server.py running? ({e})")

    # A quick live demo so you can SEE the difference with your own eyes.
    print("=== LIVE streaming demo (watch the words appear one by one) ===\n")
    run_streaming(PROMPTS[0], show_live=True)
    print("\n")

    # Warmup (skip cold-start model load).
    run_non_streaming(PROMPTS[0])

    print("=" * 70)
    print(f"{'prompt':<40} | {'mode':<13} | {'TTFT':>6} | {'total':>6}")
    print("-" * 70)
    rows = []
    for p in PROMPTS:
        ns = run_non_streaming(p)
        st = run_streaming(p)
        rows.append((p, ns, st))
        short = (p[:37] + "...") if len(p) > 40 else p
        print(f"{short:<40} | {'non-stream':<13} | {ns['ttft']:>5.2f}s | {ns['total']:>5.2f}s")
        print(f"{'':<40} | {'streaming':<13} | {st['ttft']:>5.2f}s | {st['total']:>5.2f}s")
        print("-" * 70)

    # Summary: average TTFT for each mode + how many times faster the first word felt.
    avg_ns_ttft = sum(r[1]["ttft"] for r in rows) / len(rows)
    avg_st_ttft = sum(r[2]["ttft"] for r in rows) / len(rows)
    print("\nSUMMARY")
    print(f"  avg TTFT non-streaming : {avg_ns_ttft:.2f}s  (you stare at a blank screen this long)")
    print(f"  avg TTFT streaming     : {avg_st_ttft:.2f}s  (first word appears this fast)")
    if avg_st_ttft > 0:
        print(f"  => first word appears ~{avg_ns_ttft / avg_st_ttft:.1f}x sooner with streaming")


if __name__ == "__main__":
    main()
