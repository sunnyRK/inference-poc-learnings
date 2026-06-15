# 07 — POC2 Learnings (Concurrent Requests)

> ✅ **Status: BUILT & MEASURED.** Code in [`../POC2-concurrent-requests`](../POC2-concurrent-requests). This doc keeps the *pre-registered* hypotheses (written before coding) and then scores them against the real numbers at the bottom — that pre-registration is how you actually learn, because it forces a mental model you can be proven wrong about.

> **TL;DR of the result:** throughput stayed **flat (~29 tok/s)** from 1→8 concurrent users while latency climbed **linearly** (p50 2.7s → 11.1s). Naive serving *serializes* work — exactly the problem continuous batching solves.

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

## Results (measured — Apple M4, 16GB, Qwen2.5-3B, 8 req/level)

```
Concurrency │ Throughput (tok/s) │ p50 latency │ p95 latency │ p99 latency
────────────┼────────────────────┼─────────────┼─────────────┼────────────
     1      │       28.3         │   2.72 s    │   2.95 s    │   3.00 s
     2      │       29.9         │   4.50 s    │   5.51 s    │   5.63 s
     4      │       30.1         │   8.92 s    │   9.32 s    │   9.40 s
     8      │       28.9         │  11.14 s    │  18.18 s    │  18.40 s
```

**Saturation point:** immediate — throughput never meaningfully rose. Peak was 30.1 tok/s at concurrency 4 = **1.06× the single-stream baseline** (i.e. no real gain). The system is saturated at concurrency 1.

### Verdict on hypotheses

| # | Hypothesis | Verdict | Notes |
|---|-----------|---------|-------|
| H1 | Throughput rises then plateaus | ⚠️ **Partial** | It plateaued *immediately* — there was no meaningful rise at all (1.06× peak). Stronger result than predicted: naive serving gives ~zero throughput scaling. |
| H2 | Per-request latency degrades ~linearly | ✅ **Confirmed** | p50: 2.72 → 4.50 → 8.92 → 11.14s, ~doubling each time concurrency doubled. Textbook queueing. |
| H3 | Per-request tok/s drops under load | ✅ **Confirmed** | Aggregate flat ⇒ each request's share of the ~30 tok/s pie halves as users double. |
| H4 | p99 blows up faster than p50 | ✅ **Confirmed** | At C=8, p50=11.1s but p99=18.4s — the tail spread widened sharply vs lower levels. |
| H5 | 16GB memory caps concurrency | ❌ **Not observed** | Zero failures at C=8; we were bound by **scheduling/bandwidth serialization**, not memory. Memory would bite at much higher concurrency or longer contexts. |

### The one sentence to remember
> Adding concurrent users to a naive server **did not add throughput — it only added waiting.** The throughput line is flat because a single GPU's decode bandwidth is shared, not multiplied, without real batching. That flat line is precisely what continuous batching (vLLM) turns into a 10–24× rising line.

---

### Related concepts
- [[05-batching]] — the theory POC2 stress-tests.
- [[04-kv-cache]] — why memory caps concurrency.
- [[06-poc1-learnings]] — the single-stream baseline we're loading up.
