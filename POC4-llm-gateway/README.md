# POC4 — LLM Gateway

**What it is (in 3 lines):** A reverse-proxy / API-gateway that sits **in front of** the model backend and enforces the production concerns the model server shouldn't handle itself — **API-key auth, model routing, per-key rate limiting, and usage metrics**. Same pattern as Kong / Envoy / an API gateway in front of microservices, but the upstream is an LLM. It also measures how much latency the gateway layer adds (**~0.05 ms** — basically free).

> 📚 Term meanings: [10-glossary](../notes/10-glossary.md). This POC is mostly backend/systems work, not new inference theory — it's about wrapping the model in production plumbing.

---

## Why this matters in production

A raw model server (POC1–POC3) will happily let anyone hit it, with no limits, no auth, no idea who's using what. That's fine on your laptop, fatal in production. The gateway is the **single front door** that adds:

- **Auth** — only valid API keys get in. (Stops randoms from burning your GPU budget.)
- **Routing** — clients ask for a model *alias* (`fast`, `smart`); the gateway maps it to a real backend. You can swap models/hardware behind the alias without clients changing anything.
- **Authorization (tiers)** — a free key can use `fast`; only a pro key can use `smart`. Plan enforcement.
- **Rate limiting** — cap requests per key per minute, so one user can't starve everyone else. (This is the *fairness* lever — recall [POC2](../POC2-concurrent-requests): GPU capacity is fixed, so you must ration it.)
- **Metrics** — per-key request and token counters for billing/monitoring.

This is exactly what OpenAI/Anthropic put in front of their models. The model does inference; the gateway does *everything else*.

---

## Architecture

```
                          ┌───────────────────────── GATEWAY (:8000) ─────────────────────────┐
  client ── POST ───────► │  1. AUTH        valid Bearer key?           → 401 if not           │
  Authorization:         │  2. ROUTING     alias "fast"/"smart" → real backend + config        │
   Bearer sk-...         │  3. AUTHORIZE   key allowed to use this model? → 403 if not          │
  {"model":"fast",       │  4. RATE LIMIT  under N/min for this key?    → 429 if not            │
   "message":"..."}      │  5. FORWARD ───────────────────────────────────────────┐           │
                          │  6. METRICS     count requests + tokens per key         │           │
                          └─────────────────────────────────────────────────────────┼──────────┘
                                                                                     ▼
                                                                          ┌────────────────────┐
                                                                          │  Ollama  qwen2.5:3b │
                                                                          └────────────────────┘
```

Every request passes the 5 checks in order; only requests that survive all of them reach the GPU.

---

## How to run

```bash
# 1. Start the gateway
cd POC4-llm-gateway
../venv/bin/uvicorn gateway:app --host 127.0.0.1 --port 8000

# 2. Run the full feature demo
../venv/bin/python demo.py

# Or try it by hand:
# valid call
curl -s localhost:8000/v1/chat -H 'Authorization: Bearer sk-free-001' \
  -H 'content-type: application/json' -d '{"message":"hi","model":"fast"}'
# blocked: free key can't use the "smart" model
curl -s localhost:8000/v1/chat -H 'Authorization: Bearer sk-free-001' \
  -H 'content-type: application/json' -d '{"message":"hi","model":"smart"}'
# usage stats
curl -s localhost:8000/stats
```

**Keys built in:** `sk-free-001` (free: 5/min, only `fast`) and `sk-pro-002` (pro: 60/min, `fast` + `smart`).

---

## Results (real demo output)

```
1. AUTH
  [HTTP 401] no key at all            -> Missing 'Authorization: Bearer <key>' header
  [HTTP 401] bad key 'sk-hacker'      -> Invalid API key

2. ROUTING + AUTHORIZATION
  [HTTP 200] free key + 'fast'  (allowed)      -> model=fast  key=free-tier overhead=0.06ms
  [HTTP 403] free key + 'smart' (NOT allowed)  -> Key 'free-tier' is not allowed to use model 'smart'
  [HTTP 200] pro key  + 'smart' (allowed)      -> model=smart key=pro-tier  overhead=0.05ms
  [HTTP 404] unknown 'turbo' model             -> Unknown model 'turbo'

3. RATE LIMIT (free tier = 5/min)
  request #1: HTTP 200 (OK)
  request #2: HTTP 200 (OK)
  request #3: HTTP 200 (OK)
  request #4: HTTP 200 (OK)
  request #5: HTTP 429 (REJECTED) -> Rate limit of 5/min exceeded. Retry in ~54.5s
  request #6: HTTP 429 (REJECTED)
  request #7: HTTP 429 (REJECTED)

4. STATS
  free-tier: {'ok': 5, 'rate_limited': 3, 'tokens': 50}
  pro-tier:  {'ok': 1, 'rate_limited': 0, 'tokens': 10}
```

### How to read this
- **Auth works:** no key and unknown key both get `401`. Only the two real keys pass.
- **Routing + tiers work:** the free key is allowed `fast` (`200`) but blocked from `smart` (`403`); the pro key gets `smart` (`200`); an unknown alias is `404`.
- **Rate limit works:** the free tier is capped at 5/min. *(It hit the cap on request #5 of the loop, not #6, because the free key already spent 1 request in the routing section — the sliding window counts every request in the last 60 seconds.)* Stats confirm exactly `ok=5` for free-tier.
- **Gateway overhead ≈ 0.05 ms.** All the auth/routing/rate-limit logic adds essentially nothing next to the ~2.7 s the model takes. A well-built gateway is invisible in the latency budget.

---

## Implementation notes (the engineering)

- **Rate limiter = sliding-window log.** Per key we keep a `deque` of request timestamps; on each request we evict anything older than 60 s and reject if the remaining count ≥ the limit. Simple, correct, and gives an accurate "retry in ~Xs."
- **Thread safety.** FastAPI runs sync endpoints in a threadpool, so the shared `_req_times` / `_stats` dicts are guarded by a `threading.Lock`. (Classic shared-state-under-concurrency care.)
- **Model registry = indirection.** Clients never name a real model; they name an alias. Swapping `smart` from a 3B to a 70B model, or to a different GPU pool, is a one-line config change with zero client impact.
- **Checks are ordered cheap → expensive.** Auth and routing (free) run before the rate-limit bookkeeping, and all of them run before the expensive GPU call — so rejected requests cost almost nothing.

## Honest limitations (→ future work)
- **In-memory state** — counters/limits reset on restart and don't share across replicas. Production uses **Redis** so all gateway instances see the same limits.
- **Both aliases point to the same model** (we only have qwen2.5:3b locally). The *routing mechanism* is real; the backend variety isn't.
- **No streaming through the gateway yet** — it proxies the non-streaming `/chat`. Combining the gateway with POC3 streaming (proxying SSE) is a natural next step.
- **No retries / circuit breaker / load balancing** across multiple backends — the next level of gateway maturity.

## Files
- `gateway.py` — the gateway: auth, routing, rate limit, metrics, forwarding.
- `demo.py` — exercises every feature with real HTTP calls.
- `requirements.txt` — pinned deps.
