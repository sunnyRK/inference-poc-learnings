# POC5 — Response Cache

**What it is (in 3 lines):** A read-through cache in front of the model. We hash each request into a key; if we've answered that *exact* request before, we return the stored answer from memory (a dict lookup, ~4 ms) instead of running the GPU again (~2.6 s). Same idea as caching an expensive DB query by a hash of the query — applied to LLM responses.

> 📚 Term meanings: [10-glossary](../notes/10-glossary.md). Related: this is the *exact-match* cache; [POC6](../POC6-prefix-cache) will cache *partial* (prefix) matches.

---

## Why this matters in production

Real apps ask the **same questions over and over** — FAQ bots, "summarize this doc," repeated tool prompts, popular queries. From [POC1](../POC1-local-inference-server) we know each model run costs ~2.6 s and real GPU money. Re-running the model for an identical prompt is pure waste.

A response cache turns repeated prompts into **free, instant** responses. It's the single cheapest, highest-impact optimization for any workload with repetition — and it stacks on top of everything else (streaming, gateway, batching).

```
without cache:  every request → GPU → ~2.6 s, $$$
with cache:     repeat request → memory lookup → ~0.004 s, free
```

---

## The catch: when is caching an LLM *safe*?

A cache assumes **same input → same output, every time.** That's only true if the model is **deterministic** for that input. LLMs are random by default (a "temperature" setting > 0 makes them pick different words on purpose), so two identical prompts can give two different answers — and a naive cache would wrongly freeze one random answer forever.

**Our fix:** we call the model with `temperature = 0` (greedy decoding — always pick the most likely next token). That makes the output stable for a given prompt, so caching it is correct.

**Don't cache when:**
- You *want* variety (creative writing, brainstorming) — temperature > 0 on purpose.
- The answer depends on time/freshness ("what's the news today?").
- The answer is personalized (depends on the user, not just the prompt text).
- The prompt contains private data you shouldn't store.

This safe/unsafe judgment is the real engineering lesson of POC5 — the caching code itself is easy.

---

## Architecture

```
                   ┌──────────────── CACHE SERVER (:8000) ─────────────────┐
  client ── POST ─►│  key = sha256(model | num_predict | temp | message)    │
   {"message"}     │                                                        │
                   │   in cache (and fresh)?                                 │
                   │      │                                                  │
                   │      ├── YES (HIT)  → return stored answer  (~0.004 s)  │
                   │      │                                                  │
                   │      └── NO  (MISS) → call model → store → return       │
                   └───────────────────────────────────┬────────────────────┘
                                                        │ only on a miss
                                                        ▼
                                              ┌────────────────────┐
                                              │  Ollama qwen2.5:3b  │  (~2.6 s)
                                              └────────────────────┘
```

The cache key is a **SHA-256 hash** of everything that affects the output (model, token cap, temperature, and the prompt text). If any of those change, it's a different key → a different cache slot → no false hits.

---

## How to run

```bash
# 1. Start the cache server
cd POC5-response-cache
../venv/bin/uvicorn cache_server:app --host 127.0.0.1 --port 8000

# 2. Run the benchmark
../venv/bin/python benchmark.py

# Or by hand — run the SAME prompt twice and watch "cached" flip true:
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message":"hi"}'
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"message":"hi"}'
curl -s localhost:8000/cache/stats
```

---

## Results (real, measured)

**Hardware:** Apple M4, 16 GB · **Model:** Qwen2.5-3B (Q4, temperature 0).

**A. Same prompt twice — miss vs hit:**
```
1st call: cached=False  latency=2.6661 s   (MISS -> ran the model)
2nd call: cached=True   latency=0.0043 s   (HIT  -> returned from memory)
=> the cache hit was ~616x faster
```

**B. Realistic workload (8 requests, 3 unique prompts that repeat):**
```
# 1 [MISS] 2.574 s  | What is batching in LLM inference?
# 2 [MISS] 2.671 s  | Why GPU memory matters in inference?
# 3 [MISS] 2.606 s  | What is time to first token?
# 4 [HIT ] 0.006 s  | What is batching in LLM inference?
# 5 [HIT ] 0.005 s  | Why GPU memory matters in inference?
# 6 [HIT ] 0.004 s  | What is batching in LLM inference?
# 7 [HIT ] 0.003 s  | What is time to first token?
# 8 [HIT ] 0.003 s  | What is batching in LLM inference?

requests          : 8   |   hit rate: 62.5%   (5 of 8 served free)
total WITH cache  : 7.87 s
est. WITHOUT cache: 20.94 s
=> time saved     : ~13.1 s
```

### How to read this
1. **A cache hit is ~600× faster** (≈4 ms vs ≈2.6 s). The model never runs; it's just a hash lookup in a dict.
2. **The first time each unique prompt appears it's a MISS** (you pay the full model cost once). Every repeat after that is a HIT (free).
3. **Hit rate drives the savings.** At 62.5% hit rate we did the work of 3 model runs instead of 8 — cutting total time from ~21 s to ~8 s. The higher the repetition in your traffic, the bigger the win.

**One line:** the cache doesn't make the *model* faster — it lets you **skip the model entirely** for work you've already done.

---

## Implementation notes
- **Key = SHA-256** of `model | num_predict | temp | message`. Hashing keeps keys fixed-size and includes every output-affecting field so different settings never collide.
- **`temperature=0`** on the model call makes outputs deterministic → caching is correct.
- **TTL hook** (`CACHE_TTL_SECONDS`) — entries can be set to expire, for answers that go stale. Default off here.
- **Thread-safe** — the shared `_cache` / `_stats` dicts are guarded by a `threading.Lock` (handlers run in a threadpool).
- **Endpoints:** `/chat` (cached), `/cache/stats` (hits/misses/hit-rate), `/cache/clear`.

## Honest limitations (→ future work)
- **In-memory & unbounded.** Resets on restart; grows forever. Production uses **Redis** (shared across replicas, with TTL) and an **eviction policy** (LRU — drop least-recently-used when full).
- **Exact-match only.** "What is a KV cache?" and "what is kv cache" are *different* keys → a miss, even though they mean the same thing. Fixing that needs **semantic caching** (embed the prompt, match by similarity) — a more advanced POC.
- **Only caches whole responses.** Two prompts that *share a long prefix* (e.g. the same system prompt) get no benefit here — that's exactly what [POC6 — Prefix Cache](../POC6-prefix-cache) solves at the KV-cache level.

## Files
- `cache_server.py` — the read-through cache + stats/clear endpoints.
- `benchmark.py` — miss-vs-hit speedup + a repeat-heavy workload with hit rate.
- `requirements.txt` — pinned deps.
