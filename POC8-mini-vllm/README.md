# POC8 — Mini-vLLM (own KV cache + continuous batching)

**What it is (in 3 lines):** The capstone. We stop using Ollama and run a real model (distilgpt2) directly with HuggingFace transformers, so **we** own the two things every inference engine is built around: the **KV cache** and the **batching loop**. We prove the KV cache works from the inside (3.4× faster in our own loop) and build a **continuous-batching scheduler** that scales throughput **1× → 4.5×** as we widen the batch.

> 📚 This ties together [04-kv-cache](../notes/04-kv-cache.md), [05-batching](../notes/05-batching.md) (continuous batching), [07](../POC7-kv-cache-observer) (KV proof). It's the "where do WE finally manage the KV cache" answer.

---

## Why this POC matters

Every previous POC sat *on top of* Ollama, which secretly owned the KV cache and the scheduler. To actually understand an inference engine, you have to build those two pieces yourself. POC8 does exactly that with a tiny real model on the CPU — small enough to run on a laptop, real enough that every mechanism is genuine.

This is the difference between "I use vLLM" and "I understand why vLLM is built the way it is."

---

## Part 1 — Own the KV cache (`kv_cache_demo.py`)

A manual token-by-token decode loop where **we** hold a `DynamicCache`, pass it into every forward pass, and watch it grow by one token each step. We generate the same 60 tokens two ways:

- **WITH cache** — feed only the new token each step; the cache supplies the history → O(n).
- **WITHOUT cache** — re-feed the whole sequence each step; recompute everything → O(n²).

```
$ python kv_cache_demo.py
  [with cache] after prefill, our cache holds 12 tokens
  [with cache] cache grew to 71 tokens (prompt + 60 generated)

  outputs identical?   True
  WITH cache    : 0.587 s   (102.1 tok/s)
  WITHOUT cache : 2.001 s   ( 30.0 tok/s)
  => the KV cache made our own loop ~3.4x faster
```

**This is "managing the KV cache" — in our code, not Ollama's.** POC7 proved the cache exists by measuring from the outside; here we hold the object, grow it, and feel the speedup from the inside.

---

## Part 2 — Continuous batching (`mini_engine.py`)

A scheduler that serves a queue of 8 requests with **different output lengths**. It owns:
- the **KV cache** (a shared `DynamicCache` for the active batch),
- **admission** — pull a waiting request into a free slot,
- **eviction** — drop a request the instant it finishes.

We compare serving the queue **sequentially** (one at a time) vs **continuous batching** at several batch capacities:

```
$ python mini_engine.py
  workload: 8 requests, output lengths 20-55 tokens (275 tokens total)

  mode                          time    tok/s   vs seq
  ----------------------------------------------------
  sequential (capacity=1)      6.03s    45.6     1.0x
  continuous (capacity=2)      2.81s    97.9     2.1x
  continuous (capacity=4)      1.79s   153.5     3.4x
  continuous (capacity=8)      1.34s   205.0     4.5x
```

### How to read this
- **Throughput scales with batch capacity.** 2 requests per pass ≈ 2×, 4 ≈ 3.4×, 8 ≈ 4.5×. More requests share each forward pass, so the fixed cost of a step is amortized across more work — the [note 05](../notes/05-batching.md) insight, measured in our own engine.
- **Same model, same total work (275 tokens).** Only the *scheduler* changed. We didn't make the model faster; we stopped wasting it on one request at a time.
- **It's "continuous", not static:** because requests finish at different lengths, a finished request is evicted and a waiting one admitted mid-flight — the batch stays full instead of waiting for the slowest (no head-of-line blocking).

---

## Architecture (the continuous-batching loop)

```
   queue: [r4][r5][r6][r7]          active batch (capacity 4)
                                    ┌──────────────────────────┐
                                    │ [r0][r1][r2][r3]          │  one shared KV cache
   each loop iteration:             │                          │
     1. evict finished  ───────────►│ r1 done → drop it         │
     2. admit from queue ──────────►│ r4 fills the free slot     │
     3. one forward pass over the   │ [r0][r4][r2][r3] decode +1 │
        whole active batch          └──────────────────────────┘
     4. append 1 token to each active request
```

---

## How to run

```bash
cd POC8-mini-vllm
../venv/bin/pip install -r requirements.txt   # torch + transformers (first time only)

../venv/bin/python kv_cache_demo.py   # Part 1: own the KV cache
../venv/bin/python mini_engine.py     # Part 2: continuous batching
```
First run downloads distilgpt2 (~330 MB) and loads it (~15 s); after that it's cached. Runs on CPU — no GPU needed.

---

## Honest limitations (this is a *mini* vLLM)

- **We re-prefill on every batch change.** When a request joins or leaves, we rebuild the shared KV cache by re-processing the active sequences. That's wasteful — and it's *exactly* the problem **PagedAttention** (vLLM) solves: it stores the KV cache in fixed-size pages so sequences can be added/removed without recomputing. Our `rebuilds` counter shows how often we pay that cost.
- **CPU + a tiny 82M model.** Real gains are far bigger on a GPU, where decode is memory-bandwidth-bound and batching is nearly free. The *shape* (throughput rises with batch size) is the transferable lesson, not the absolute tok/s.
- **No PagedAttention, no prefix sharing, no fancy scheduling policy.** Those are the next layers of a production engine.
- **Greedy decoding only**, fixed lengths. No sampling, no streaming (POC3 covered streaming; combining them is straightforward).

## What this POC earns you (résumé-ready)
*"Built a mini inference engine from scratch (PyTorch + HF transformers): a manual KV-cache decode loop and a continuous-batching scheduler with admission/eviction over a shared cache, demonstrating throughput scaling to ~4.5× vs sequential serving — the core mechanism behind vLLM/TGI."*

## Files
- `kv_cache_demo.py` — own the KV cache; with-vs-without cache speedup.
- `mini_engine.py` — sequential vs continuous batching, throughput-vs-capacity sweep.
- `requirements.txt` — torch + transformers.
