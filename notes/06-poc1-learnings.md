# 06 — POC1 Learnings (Local Inference Server)

> What our naive baseline actually taught us — backed by real numbers measured on this machine, not theory.

## Setup (reproducible)

| | |
|---|---|
| **Hardware** | Apple M4, 16 GB unified memory |
| **Model** | Qwen2.5-3B (Q4 quantized, ~1.9 GB on disk) |
| **Engine** | Ollama (llama.cpp + Metal backend) |
| **Server** | FastAPI + Uvicorn, single worker |
| **Output cap** | `num_predict: 80` tokens |
| **Mode** | Sequential — one request at a time (no concurrency, no streaming) |

## The raw numbers

```
Prompt                              Latency   Out tokens   Tokens/sec
─────────────────────────────────────────────────────────────────────
Explain KV cache in simple way      5.673 s      80          32.87   ← FIRST (cold)
What is batching in LLM inference?  2.772 s      80          31.93
Explain tokenization in 2 lines     1.265 s      33          34.21
What is time to first token?        2.880 s      80          30.50
Why GPU memory matters?             2.674 s      80          33.35
─────────────────────────────────────────────────────────────────────
Avg tokens/sec ≈ 32.6   |   Warm 80-tok latency ≈ 2.78 s
```

---

## Learning 1 — Cold start is real and it's huge

The **first** request took **5.67s**; identical-shaped requests afterward took **~2.7s**. That ~3s gap isn't compute — it's **loading the 1.9 GB model from disk into memory** on first use. The model wasn't resident yet.

**Production implication:** this is the *cold-start problem* that dominates serverless GPU inference. When a model isn't loaded, the first user eats the load time. Real systems fight this with:
- **Model pinning / keep-warm** (don't evict the model between requests).
- **Provisioned/min replicas** so a warm instance always exists.
- **Fast weight loading** (mmap, safetensors, tensor parallelism for load).

> Blockchain analogy: it's a **cold cache vs warm cache** on node startup — the first query after boot pays to load state into memory; subsequent ones are fast.

**Takeaway:** always discard the first measurement (warmup run) when benchmarking, *and* track cold start separately because users feel it.

---

## Learning 2 — Tokens/sec measures the *engine*; latency measures the *request*

Tokens/sec stayed in a tight band (**30.5–34.2**) regardless of prompt or answer length, while latency swung from **1.27s to 5.67s**. They measure different things:

- **tokens/sec** = `eval_count / eval_duration` → pure **decode speed**, a property of (model × hardware). On an M4 with a 3B Q4 model, ~32 tok/s is our ceiling. It barely moves.
- **latency** = queue + prefill + **decode** + network/JSON → a property of the **whole request**, dominated by *how many tokens you generate*.

The 33-token answer (1.27s) vs the 80-token answers (~2.7s) proves it: **same engine speed, ~2.4× the tokens, ~2.2× the latency.** Output length is the latency dial.

---

## Learning 3 — You can decompose the latency

For a warm 80-token request at ~32 tok/s:

```
pure decode time   = 80 tokens ÷ 32 tok/s   ≈ 2.50 s
measured latency                            ≈ 2.78 s
─────────────────────────────────────────────────────
overhead (prefill + HTTP + JSON + Python)   ≈ 0.28 s   (~10%)
```

For the cold request:

```
pure decode time   = 80 ÷ 32.87            ≈ 2.43 s
measured latency                           ≈ 5.67 s
─────────────────────────────────────────────────────
overhead (mostly model load)               ≈ 3.24 s   ← the cold-start tax
```

**Takeaway:** decode dominates warm latency (~90%), but you can still lose 10% to the serving layer — and *all* of it to a cold start. Knowing the breakdown tells you where optimization pays off. (Streaming, POC3, attacks the *perceived* latency by delivering tokens as they're produced instead of waiting for all 80.)

---

## Learning 4 — The naive server wastes the GPU

This server handles exactly **one request at a time**. While Qwen is decoding, a second user is fully blocked; while we wait on the network, the GPU is idle. From [[05-batching]] we know decode is memory-bound and the GPU sits ~5–10% utilized for a single stream. **We are paying for a GPU and using a fraction of it.** That waste is the entire motivation for POC2 onward.

---

## Learning 5 — Measurement is a feature, not an afterthought

The most valuable thing POC1 does isn't serving — it's **returning `latency`, `prompt_eval_count`, `eval_count`, and `tokens_per_second` on every response.** You cannot optimize what you cannot see. Every serious inference stack (vLLM's `/metrics`, OpenAI's `usage` object) exposes exactly these. Building the instrumentation first means every later POC has a baseline to beat.

---

## What POC1 deliberately does NOT do (the roadmap writes itself)

| Limitation in POC1 | The POC that fixes it | Concept |
|---|---|---|
| Waits for all 80 tokens before responding | **POC3** Streaming | TTFT / perceived latency |
| One request at a time | **POC2** Concurrency | throughput, [[05-batching]] |
| Recomputes identical prompts every time | **POC5** Response cache | exact-match caching |
| Recomputes shared prompt prefixes | **POC6** Prefix cache | [[04-kv-cache]] sharing |
| KV cache is invisible | **POC7** KV visualizer | memory mental model |
| Relies on Ollama's scheduler | **POC8** Mini-vLLM | continuous batching |

---

## Concepts this POC made concrete

- [[01-what-is-inference]] — we *measured* prefill vs decode by their fingerprints (TTFT-ish overhead vs tok/s).
- [[02-tokenization]] — `eval_count`/`prompt_eval_count` are the token counters that drive cost and latency.
- [[04-kv-cache]] — stable tok/s across an 80-token response is the KV cache working silently underneath Ollama.
- [[05-batching]] — single-stream serving is the baseline that batching will demolish.

---

### One-line résumé bullet this POC earns
*"Built an instrumented local LLM inference server (FastAPI + Ollama/Qwen2.5) and benchmarked TTFT, tokens/sec, and cold-start latency on Apple Silicon; established a measurement baseline for evaluating batching, caching, and streaming optimizations."*

Next: [[07-poc2-learnings]] — putting concurrent load on this baseline.
