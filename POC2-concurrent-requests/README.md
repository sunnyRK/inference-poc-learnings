# POC2 — Concurrent Request Benchmark

**What it is (in 3 lines):** POC1 served one request at a time. POC2 fires many requests at that same server *simultaneously* and measures what happens to throughput and latency as load rises. The result proves — with our own numbers — that naive single-machine serving **doesn't scale**, which is the whole reason continuous-batching engines like vLLM exist.

> 📚 Theory: [05-batching](../notes/05-batching.md) · [04-kv-cache](../notes/04-kv-cache.md). Full analysis + hypothesis verdicts: [07-poc2-learnings](../notes/07-poc2-learnings.md).

---

## Why this POC matters

A model that's fast for one user but collapses under 50 concurrent users can't ship. The metrics that matter flip from *per-request* (latency) to *aggregate* (throughput, tail latency). This POC measures the **degradation curve** — how p50/p95/p99 latency and tokens/sec change as you add concurrent users — and finds the **saturation point** where adding users stops adding throughput and only adds waiting.

---

## Architecture

```
   load_test.py                              POC1 server                  Ollama
 ┌───────────────────┐                    ┌──────────────┐           ┌──────────────┐
 │ ThreadPoolExecutor│  C concurrent      │  FastAPI      │  C block- │ qwen2.5:3b   │
 │  max_workers = C  │  POST /chat        │  (threadpool) │  ing POSTs│ ONE GPU,     │
 │  ──► fire C reqs  │ ─────────────────► │  ──► forward  │ ────────► │ shared decode│
 │      at once      │                    │     to Ollama │           │ bandwidth    │
 │  measure each     │ ◄───────────────── │              │ ◄──────── │              │
 │  latency + tokens │   JSON responses   │              │           │              │
 └───────────────────┘                    └──────────────┘           └──────────────┘
       client side                          our code                  the real engine
```

We use a **thread pool**, not asyncio: `requests` is blocking but releases the GIL while waiting on the socket, so `max_workers=C` keeps exactly **C requests in flight** at the server. Extra requests queue on the client. Zero new dependencies.

---

## How to run

```bash
# 1. Start the POC1 server (separate terminal)
cd ../POC1-local-inference-server
../venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# 2. Run the load test (this folder)
cd ../POC2-concurrent-requests
../venv/bin/python load_test.py

# custom sweep:
../venv/bin/python load_test.py --levels 1,2,4,8,16 --requests 12
```

`--levels` = concurrency levels to sweep. `--requests` = total requests fired at each level. A warmup request runs first so the cold-start model-load tax doesn't pollute level 1.

---

## Results (real, measured)

**Hardware:** Apple M4, 16 GB · **Model:** Qwen2.5-3B (Q4) · **Engine:** Ollama · 8 requests per level, `num_predict=80`.

```
conc |   req/s |   tok/s |  p50 (s) |  p95 (s) |  p99 (s)
-----+---------+---------+----------+----------+---------
   1 |    0.42 |    28.3 |     2.72 |     2.95 |     3.00
   2 |    0.45 |    29.9 |     4.50 |     5.51 |     5.63
   4 |    0.44 |    30.1 |     8.92 |     9.32 |     9.40
   8 |    0.43 |    28.9 |    11.14 |    18.18 |    18.40
-----+---------+---------+----------+----------+---------
Peak throughput: 30.1 tok/s at concurrency 4 = 1.06x the single-stream baseline
```

### The two lines that tell the whole story

```
THROUGHPUT (tok/s)              LATENCY p50 (s)
~30 ┤●────●────●────●           11 ┤              ●
    │  FLAT — no gain            9 ┤         ●
    │                            5 ┤    ●
  0 ┼────────────────           3 ┤●
    1    2    4    8                1    2    4    8
        concurrency                    concurrency
   throughput stays flat        latency climbs ~linearly
```

---

## What this means (the punchline)

1. **Throughput is flat (~29 tok/s) no matter how many users.** Going from 1 → 8 concurrent users did **not** increase total work done. Peak was 1.06× the baseline — basically noise.

2. **Latency degrades almost perfectly linearly.** p50 went 2.72 → 4.50 → 8.92 → 11.14s as concurrency doubled each step. Each user's wait grows in direct proportion to how many users are ahead of them.

3. **The tail (p99) blows up fastest.** At concurrency 8, p50 was 11.1s but p99 hit **18.4s** — the unlucky requests suffer far more than the median. Tail latency always degrades first and worst under contention.

**Diagnosis:** this is the signature of **no real batching** — the work is being *serialized*. A single GPU has a fixed decode bandwidth (~30 tok/s for this model), and Ollama's default scheduling shares it across requests rather than truly batching them. So N concurrent users just split the same pie: aggregate throughput stays flat while each user waits N× longer. This is **head-of-line blocking / queueing**, exactly as predicted in [05-batching](../notes/05-batching.md).

**Why it matters:** this flat line is the problem **continuous batching** solves. A real engine (vLLM, TGI) reschedules the batch every decode token, packing many sequences through the GPU together so the throughput line *rises* (10–24×) instead of staying flat — until it hits the genuine memory-bandwidth ceiling. POC8 (mini-vLLM) will rebuild that, and the eventual vLLM POC will rerun *this exact load test* to quantify the difference.

---

## Limitations & honest caveats

- **Single small model on a laptop GPU.** The ~30 tok/s ceiling is this machine's memory bandwidth for a 3B model. The *shape* of the curve (flat throughput, linear latency) is the transferable lesson, not the absolute numbers.
- **Ollama isn't a continuous-batching engine.** Raising `OLLAMA_NUM_PARALLEL` lets it run a few sequences concurrently, but on one Metal GPU they still share decode bandwidth, so aggregate throughput is bandwidth-bound regardless. That's *why* we'll move to vLLM later.
- **Client and server on the same machine.** Real deployments separate them; here the load generator competes for the same CPU, which slightly affects absolute timings (not the trend).

---

## Files
- `load_test.py` — threaded load generator; sweeps concurrency levels, reports throughput + p50/p95/p99.
- `requirements.txt` — just `requests`.
