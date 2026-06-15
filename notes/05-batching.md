# 05 — Batching

> Batching is how one expensive GPU serves many users at once. It's the difference between an inference server that costs $100k/month and one that costs $10k/month for the same traffic. This is the highest-leverage systems idea in inference serving.

## Why batching is almost free (the key insight)

From [[01-what-is-inference]]: during **decode**, the bottleneck is **reading the model weights from GPU memory**, not the math. To generate one token for one user, you stream ~14 GB of weights through the chip and do a tiny amount of compute with them. The compute units sit mostly idle — you're memory-bandwidth-bound.

Here's the magic: **once those weights are loaded onto the chip, applying them to 1 token or 32 tokens costs almost the same time.** The expensive part (moving weights) is shared; the cheap part (the matmul) scales.

```
Serve 1 user:    read 14GB weights ──► compute 1 token   ← weights cost dominates, GPU 5% utilized
Serve 32 users:  read 14GB weights ──► compute 32 tokens ← same weight read, GPU 80% utilized
```

So batching 32 requests gives you **~32× throughput for ~1× the cost.** This is why batching is *the* throughput lever.

> Backend analogy: it's **request coalescing / vectorized I/O**. Like batching 100 row-inserts into one DB round-trip, or coalescing many small reads into one page fetch — you amortize a fixed expensive cost (weight movement / round-trip) over many units of work.

---

## The tradeoff: throughput vs latency

Batching isn't pure win. It trades **latency for throughput**:

- Bigger batch → higher throughput (tokens/sec across all users) ✅
- Bigger batch → each individual user may wait longer (queueing + larger per-step compute) ❌
- A user who arrives just after a batch starts may wait for it to make progress.

The whole job of an inference scheduler is to **maximize throughput while keeping per-user latency under the SLA** (this is "goodput" from [[01-what-is-inference]]). Every engine is a different answer to this scheduling problem.

---

## The evolution of batching (this is the actual history)

### 1. No batching (POC1)
One request at a time. GPU idle between requests. Simple, terrible utilization. **This is where we are now.**

### 2. Static / dynamic batching
Wait a few ms, collect requests that arrived, run them as one batch.

```
   req A ─┐
   req B ─┼─►  [ batch of 3 ]  ──► run together ──► all finish together
   req C ─┘
```

**The fatal flaw:** requests have *different output lengths*. If A needs 10 tokens and B needs 500, the whole batch is held hostage until B finishes. A's user waits 50× too long, and the batch slot sits half-empty. This is **head-of-line blocking**, and it's exactly the kind of bug a backend engineer recognizes instantly.

```
batch: [A: 10 tokens ][~~~~~~ A done, slot WASTED ~~~~~~]
       [B: 500 tokens ──────────────────────────────────]
                       ↑ A's user is done but can't leave; new users can't enter
```

### 3. Continuous batching (a.k.a. in-flight batching) — the breakthrough

The insight: a batch isn't a fixed group — it's a **rolling set of slots**, managed **per decode step**. Because generation is token-by-token (one forward pass per token), you can change the batch *between every single token*:

```
step t:    [A][B][C][D]   ← 4 slots busy
A finishes ──► its slot frees immediately
step t+1:  [E][B][C][D]   ← new request E slotted in instantly, no waiting
```

- A finishes? Evict it *this step*, admit a waiting request into its slot.
- No request waits for an unrelated long one.
- The GPU stays near-fully packed every step → maximum utilization.

This is **the** core innovation of vLLM (combined with PagedAttention for the memory side) and what TGI calls "in-flight batching." It's why these engines deliver 10–24× the throughput of naive serving. **If you understand head-of-line blocking and connection pooling, you already understand continuous batching — it's the same idea applied to GPU decode steps.**

---

## Prefill vs decode batching (the subtlety)

Prefill and decode have opposite profiles ([[01-what-is-inference]]), so batching them together is awkward:

- **Prefill** is compute-bound and bursty (one big matmul over a whole prompt).
- **Decode** is memory-bound and steady (one token per step).

Mixing a big new prefill into a decode batch causes a latency spike for the users mid-generation (their next token gets delayed behind the prefill). Modern engines handle this with:

- **Chunked prefill** — split a long prompt's prefill into smaller chunks interleaved with ongoing decode, so no single step is dominated by a giant prefill.
- **Disaggregated serving** — run prefill and decode on *separate* GPU pools entirely (used at scale by frontier labs).

You don't need these yet, but knowing the terms is interview gold.

---

## Why POC1 leaves all this on the table

POC1 serves **one request at a time** — zero batching, GPU mostly idle. That's intentional: it's the baseline. The numbers (~30–34 tok/s, single stream) are the *floor*.

**POC2 (concurrent requests)** is the first time we put multiple requests in flight and watch what happens:
- Does total throughput go up (good batching) or does latency just degrade linearly (no real batching)?
- Ollama/llama.cpp has limited batching vs vLLM — POC2 will *measure* that ceiling and motivate why we eventually need a real engine.

By POC8 (mini-vLLM) we'll implement a toy **continuous batching** loop ourselves and watch throughput jump.

---

## The numbers that make managers care

| Strategy | Relative throughput | Per-user latency | GPU utilization |
|---|---|---|---|
| No batching (POC1) | 1× | best (no queue) | ~5–10% |
| Static batching | 3–5× | spiky (HoL blocking) | medium |
| **Continuous batching** (vLLM) | **10–24×** | good & stable | ~70–90% |

Same GPU, same model, same electricity bill. The only difference is **scheduling software**. That software is the job you're training for.

---

### Interview-ready summary
- Batching amortizes the dominant cost (weight movement) over many requests → near-linear throughput gain.
- Static batching suffers head-of-line blocking from uneven output lengths.
- **Continuous batching** reschedules the batch every decode token, evicting finished requests and admitting new ones → the core of vLLM/TGI.
- Prefill vs decode have opposite profiles → chunked/disaggregated prefill manage the mix.

Next: [[06-poc1-learnings]] — what our naive baseline actually measured, with real M4 numbers.
