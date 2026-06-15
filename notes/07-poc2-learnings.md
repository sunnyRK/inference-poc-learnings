# 07 — POC2 Plan & Hypotheses (Concurrent Requests)

> ⏳ **Status: NOT YET BUILT.** This is the design + hypotheses doc we write *before* coding, so we can compare predictions against reality. Results will be filled in when we build POC2 together. (Writing hypotheses first is how you actually learn — it forces a mental model you can be wrong about.)

## The question POC2 answers

POC1 served **one request at a time**. The obvious next question for any backend engineer:

> *"What happens when 10, 20, 50 users hit this server simultaneously?"*

Does throughput scale up (the server batches work)? Or does every user just get proportionally slower (no real concurrency)? **We're going to measure the ceiling of naive serving** — and that measurement is what motivates everything after it.

---

## Why this matters in production

This is the difference between a demo and a product. A model that's fast for one user but collapses under 50 concurrent users can't ship. The metrics that matter flip from *per-request* to *aggregate*:

- **Throughput** (requests/sec, total tokens/sec) — how much the whole server does.
- **Latency under load** — p50/p95/p99, not just the average. Tail latency is what users actually complain about.
- **Latency degradation curve** — how p95 grows as concurrency rises. The *shape* of this curve reveals the bottleneck.
- **Saturation point** — the concurrency level where adding users stops adding throughput and only adds latency.

---

## Architecture to build

```
                 ┌──────────────────────────────────────────┐
   N workers ──► │  load generator (asyncio / threads)       │
   fire requests │  - send N concurrent POST /chat           │
   at once       │  - record per-request start/end + tokens  │
                 └───────────────────┬──────────────────────┘
                                     │  N concurrent
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │  FastAPI server (POC1, reused)            │
                 │  → forwards to Ollama (one model loaded)  │
                 └───────────────────┬──────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │  Ollama / llama.cpp                       │
                 │  (limited parallelism — the real ceiling) │
                 └──────────────────────────────────────────┘
```

Then sweep concurrency = 1, 2, 4, 8, 16, 32 and plot throughput & p95 latency vs concurrency.

### Implementation notes to get right
- **Use `asyncio` + `httpx.AsyncClient`** (or a thread pool) for the load generator — `requests` in a loop is sequential and would measure nothing.
- **Watch the FastAPI side too:** our POC1 `chat` handler is a sync `def`, so FastAPI runs it in a threadpool. The blocking `requests.post` to Ollama means real concurrency is gated by (a) FastAPI's threadpool and (b) Ollama's own parallelism (`OLLAMA_NUM_PARALLEL`). Part of POC2 is *finding which one is the bottleneck.*
- **Measure aggregate tokens/sec**, not just request count — that's the true throughput.
- **Report percentiles**, not just mean. Compute p50/p95/p99 of latency.

---

## Hypotheses (to be proven right or wrong)

1. **H1 — Throughput will rise then plateau.** Going from 1→4 concurrent, total tokens/sec should increase (Ollama has *some* batching/parallelism). Beyond some point it flattens — the saturation point. *Prediction: plateau somewhere low (4–8), because llama.cpp on a single M4 isn't a real continuous-batching engine.*

2. **H2 — Per-request latency degrades roughly linearly past saturation.** Once saturated, doubling concurrency ≈ doubling each user's latency (requests queue). This is the head-of-line / no-real-batching signature from [[05-batching]].

3. **H3 — tokens/sec *per request* drops under load** even though *aggregate* may rise — the GPU's fixed decode bandwidth gets split across requests.

4. **H4 — p99 will blow up faster than p50.** Tail latency always degrades first and worst under contention.

5. **H5 — The 16 GB memory limit will cap concurrency** before compute does, because each concurrent request needs its own KV cache ([[04-kv-cache]]).

---

## What POC2 sets up for the rest of the series

If the hypotheses hold, POC2 *proves by measurement* that naive serving doesn't scale — which is the entire reason real inference engines exist. That's the perfect on-ramp to:

- **POC3 (streaming)** — improve *perceived* latency even while throughput is capped.
- **POC8 (mini-vLLM)** — implement continuous batching ourselves and watch the plateau move up.
- The eventual **vLLM production POC** — run the same load test against vLLM and quantify the 10–24× throughput difference from [[05-batching]] *with our own numbers.*

---

## Results (to be filled in)

```
Concurrency │ Throughput (tok/s) │ p50 latency │ p95 latency │ p99 latency
────────────┼────────────────────┼─────────────┼─────────────┼────────────
     1      │        TBD         │     TBD     │     TBD     │    TBD
     2      │        TBD         │     TBD     │     TBD     │    TBD
     4      │        TBD         │     TBD     │     TBD     │    TBD
     8      │        TBD         │     TBD     │     TBD     │    TBD
    16      │        TBD         │     TBD     │     TBD     │    TBD
    32      │        TBD         │     TBD     │     TBD     │    TBD
```

**Saturation point:** _TBD_
**Verdict on hypotheses:** _TBD_

---

### Related concepts
- [[05-batching]] — the theory POC2 stress-tests.
- [[04-kv-cache]] — why memory caps concurrency.
- [[06-poc1-learnings]] — the single-stream baseline we're loading up.
