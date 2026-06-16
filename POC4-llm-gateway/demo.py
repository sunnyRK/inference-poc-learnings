"""
POC4 — Gateway demo.

Walks through every gateway feature with real HTTP calls and prints the
status code + result, so you can see auth, routing, authorization, and
rate limiting actually working.

Run (with gateway.py running on port 8000):
    python demo.py
"""

import requests

BASE = "http://localhost:8000"
FREE_KEY = "sk-free-001"  # free tier: 5 req/min, only "fast" model
PRO_KEY = "sk-pro-002"    # pro tier:  60 req/min, "fast" + "smart" models


def call(message: str, key: str | None = None, model: str = "fast"):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.post(f"{BASE}/v1/chat", json={"message": message, "model": model}, headers=headers)
    try:
        return r.status_code, r.json()
    except Exception:  # noqa: BLE001
        return r.status_code, {"detail": r.text}


def show(label: str, status: int, body: dict):
    line = f"[HTTP {status}] {label}"
    if status == 200:
        line += (f"  -> model={body['model']} key={body['key']} "
                 f"tokens={body['eval_count']} overhead={body['gateway_overhead_ms']}ms")
    else:
        line += f"  -> {body.get('detail')}"
    print(line)


def main():
    try:
        requests.get(f"{BASE}/", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Gateway not reachable at {BASE} — is gateway.py running? ({e})")

    msg = "Say hello in one short sentence."

    print("\n--- 1. AUTH: requests without a valid key are rejected ---")
    show("no key at all", *call(msg, key=None))
    show("bad key 'sk-hacker'", *call(msg, key="sk-hacker"))

    print("\n--- 2. ROUTING + AUTHORIZATION: keys can only use allowed models ---")
    show("free key + 'fast' model (allowed)", *call(msg, key=FREE_KEY, model="fast"))
    show("free key + 'smart' model (NOT allowed)", *call(msg, key=FREE_KEY, model="smart"))
    show("pro key + 'smart' model (allowed)", *call(msg, key=PRO_KEY, model="smart"))
    show("any key + unknown 'turbo' model", *call(msg, key=PRO_KEY, model="turbo"))

    print("\n--- 3. RATE LIMIT: free tier is 5/min; fire 7 fast and watch 429s ---")
    for i in range(1, 8):
        status, body = call(msg, key=FREE_KEY, model="fast")
        tag = "OK" if status == 200 else "REJECTED"
        print(f"  request #{i}: HTTP {status} ({tag})"
              + ("" if status == 200 else f" -> {body.get('detail')}"))

    print("\n--- 4. STATS: per-key usage counters ---")
    stats = requests.get(f"{BASE}/stats").json()
    for tier, counters in stats.items():
        print(f"  {tier}: {counters}")


if __name__ == "__main__":
    main()
