"""
POC6 — Prefix Cache benchmark.

Sends the SAME long system prompt (the reusable prefix) with several DIFFERENT
questions, and watches the prefill time collapse after the first request —
because the model reuses the prefix's KV cache instead of recomputing it.

Key metric: prefill_ms (= time to process the prompt before generating,
basically TTFT). Lower = the shared prefix was reused.

Run (with prefix_server.py running on port 8000):
    python benchmark.py
"""

import requests

BASE = "http://localhost:8000"

# A long, realistic system prompt = the expensive prefix we want to reuse.
LONG_SYSTEM = (
    "You are an expert senior software engineer and patient teacher. "
    "Always answer correctly, concisely, and with concrete examples. "
    "Follow the user's instructions carefully and stay on topic. "
) * 12

QUESTIONS = ["What is Python?", "What is Rust?", "What is Go?",
             "What is Java?", "What is C++?"]


def ask(system: str, question: str) -> dict:
    return requests.post(f"{BASE}/chat", json={"system_prompt": system, "question": question}).json()


def main():
    try:
        requests.get(f"{BASE}/", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Server not reachable at {BASE} — is prefix_server.py running? ({e})")

    print("Same long system prompt (the PREFIX) + 5 different questions (the SUFFIX).")
    print("Watch prefill_ms drop after the first request — the prefix KV cache is reused.\n")
    print(f"{'#':>2}  {'state':<5}  {'prefill':>9}  {'speedup':>8}   question")
    print("-" * 60)
    for i, q in enumerate(QUESTIONS, 1):
        d = ask(LONG_SYSTEM, q)
        state = "WARM" if d["prefix_warm"] else "COLD"
        print(f"{i:>2}  {state:<5}  {d['prefill_ms']:>7.1f}ms  {d['prefill_speedup_vs_cold']:>6}x   {q}")

    print("\nNow a DIFFERENT system prompt (a brand-new prefix) — cold again:")
    d = ask("You are a pirate. Answer in pirate speak.", "What is Go?")
    state = "WARM" if d["prefix_warm"] else "COLD"
    print(f"    {state:<5}  {d['prefill_ms']:>7.1f}ms   (new prefix, nothing to reuse)")

    stats = requests.get(f"{BASE}/prefix/stats").json()
    print("\nPREFIX REGISTRY")
    for pid, info in stats.items():
        print(f"  {pid}  hits={info['hits']:<2} cold_prefill={info['cold_prefill_ms']}ms "
              f"tokens={info['prompt_tokens']}  prefix={info['preview']!r}")


if __name__ == "__main__":
    main()
