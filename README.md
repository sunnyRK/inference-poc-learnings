# Inference Engineering — POC Learnings

> Building up LLM **inference & serving** infrastructure from first principles — one focused proof-of-concept at a time. Theory → implementation → benchmarks → production concepts.

I'm a senior backend & blockchain engineer moving into **AI infrastructure**. Rather than only consuming AI APIs, this repo is me learning the hard parts of *serving* LLMs in production — latency, throughput, batching, KV cache, GPU memory, distributed inference — by **building and benchmarking** each idea, then connecting it back to how real engines (vLLM, SGLang, TensorRT-LLM, Triton) work.

Every POC ships with: **runnable code · README · architecture diagram · real benchmarks · written learnings · tradeoffs · what's next.**

---

## 📊 Progress

| # | POC | Concept | Status |
|---|-----|---------|--------|
| 1 | [Local Inference Server](./POC1-local-inference-server) | Serving basics, instrumentation, latency/throughput measurement | ✅ Done |
| 2 | [Concurrent Requests](./POC2-concurrent-requests) | Throughput, concurrency, saturation, tail latency | ✅ Done |
| 3 | [Streaming Inference](./POC3-streaming-inference) | TTFT, token streaming (SSE), perceived latency | ✅ Done |
| 4 | [LLM Gateway](./POC4-llm-gateway) | API-key auth, model routing, rate limiting, metrics | ✅ Done |
| 5 | [Response Cache](./POC5-response-cache) | Exact-match caching, cache-hit speedup, cost reduction | ✅ Done |
| 6 | [Prefix Cache](./POC6-prefix-cache) | KV-cache reuse for shared prefixes, RadixAttention idea | ✅ Done |
| 7 | [KV Cache Observer](./POC7-kv-cache-observer) | Proving the KV cache works via prefill-vs-decode | ✅ Done |
| 8 | [Mini vLLM](./POC8-mini-vllm) | Own KV cache + continuous batching (real model) | ✅ Done |
| 9 | Benchmark Lab | Standardized load testing across engines | ⬜ Planned |
| + | Quantization · RAG serving · GPU deploy · vLLM prod · Multi-GPU · TensorRT/SGLang · Triton kernels · FlashAttention | Advanced infra | ⬜ Backlog |

👉 Full 20-POC plan with status: **[ROADMAP.md](./ROADMAP.md)**

---

## 📚 Theory notes

Concept write-ups (mentor-style, with diagrams and production connections). Read these alongside the POCs:

| Note | What it covers |
|------|----------------|
| [01 — What is Inference](./notes/01-what-is-inference.md) | Prefill vs decode, TTFT/throughput vocabulary, the engine landscape |
| [02 — Tokenization](./notes/02-tokenization.md) | BPE, why tokens drive cost/latency/context |
| [03 — Attention](./notes/03-attention.md) | Q/K/V, causal masking, the O(n²) cost center, GQA/MQA |
| [04 — KV Cache](./notes/04-kv-cache.md) | The optimization that makes generation affordable; PagedAttention & prefix caching |
| [05 — Batching](./notes/05-batching.md) | Why batching is ~free, static vs **continuous batching** |
| [06 — POC1 Learnings](./notes/06-poc1-learnings.md) | Real M4 numbers: cold start, decode ceiling, latency decomposition |
| [07 — POC2 Learnings](./notes/07-poc2-learnings.md) | Concurrency results: flat throughput, linear latency, tail blowup |
| [08 — Understanding the Numbers](./notes/08-understanding-the-numbers.md) | Beginner guide: latency, throughput, p50/p95/p99 — with simple math |
| [09 — Streaming & TTFT](./notes/09-streaming-and-ttft.md) | Beginner guide: why streaming feels ~9x faster (Time To First Token) |
| [10 — Glossary](./notes/10-glossary.md) | Plain-English definitions of inference terms, mapped to backend ideas |
| [11 — Proving the KV Cache](./notes/11-proving-kv-cache.md) | How POC7 turns "the KV cache exists" into a measured fact |
| [12 — Continuous Batching](./notes/12-continuous-batching.md) | The vLLM core idea: keep the batch full every token (POC8) |

---

## 🏗️ Repo structure

```
inference-poc-learnings/
├── README.md                       ← you are here (portfolio overview)
├── notes/                          ← theory + per-POC learnings
│   ├── 01-what-is-inference.md
│   ├── 02-tokenization.md
│   ├── 03-attention.md
│   ├── 04-kv-cache.md
│   ├── 05-batching.md
│   ├── 06-poc1-learnings.md
│   └── 07-poc2-learnings.md
├── POC1-local-inference-server/    ← FastAPI + Ollama, instrumented
│   ├── main.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── README.md
├── POC2-concurrent-requests/       ← threaded load test, p50/p95/p99
│   ├── load_test.py
│   ├── requirements.txt
│   └── README.md
├── POC3-streaming-inference/       ← token streaming (SSE), TTFT
│   ├── stream_server.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── README.md
├── POC4-llm-gateway/               ← auth, routing, rate limit, metrics
│   ├── gateway.py
│   ├── demo.py
│   ├── requirements.txt
│   └── README.md
├── POC5-response-cache/            ← exact-match response cache (read-through)
│   ├── cache_server.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── README.md
├── POC6-prefix-cache/              ← shared-prefix KV reuse + radix tree
│   ├── radix_tree.py
│   ├── prefix_server.py
│   ├── benchmark.py
│   ├── requirements.txt
│   └── README.md
├── POC7-kv-cache-observer/         ← prove KV cache via prefill-vs-decode
│   ├── kv_observer.py
│   ├── requirements.txt
│   └── README.md
├── POC8-mini-vllm/                 ← own KV cache + continuous batching (torch)
│   ├── kv_cache_demo.py
│   ├── mini_engine.py
│   ├── requirements.txt
│   └── README.md
└── (POC9, POC10, ... added as built)
```

Each `POCx-*/` folder is self-contained: its own README, code, deps, and benchmark.

---

## 🔑 Highlights so far

**POC1 — baseline:**
- Built an **instrumented** FastAPI inference server over Ollama/Qwen2.5-3B that reports latency, prompt/output token counts, and tokens/sec on every request.
- Measured on Apple M4: **~32 tok/s decode ceiling**, **~3s cold-start tax**, and confirmed **latency scales linearly with output tokens** while tokens/sec stays flat (the KV cache working underneath).
- Decomposed warm latency into **~90% decode + ~10% serving overhead**.

**POC2 — concurrency:**
- Built a threaded load generator and swept concurrency 1→8, reporting throughput + p50/p95/p99.
- **Measured the failure mode of naive serving:** throughput stayed **flat (~29 tok/s)** while latency climbed **linearly** (p50 2.7s → 11.1s) and the **tail blew up** (p99 18.4s at C=8).
- Pre-registered 5 hypotheses before coding and scored them against reality — proving *why* continuous-batching engines (vLLM) exist, with my own numbers.

**POC3 — streaming:**
- Added a streaming endpoint (Server-Sent Events, OpenAI-style `data:` format) that emits each token as the model writes it.
- **Cut Time-To-First-Token from 2.94s → 0.31s (~9.5× sooner)** while total time stayed ~3s — proving streaming improves *perceived* latency for free, even when raw speed can't change.

**POC4 — gateway:**
- Built an API-gateway in front of the model: **Bearer-key auth, model-alias routing, tier-based authorization, per-key sliding-window rate limiting, and usage metrics**.
- Verified all paths (401 / 403 / 404 / 429 / 200) and measured **gateway overhead at ~0.05 ms** — production plumbing that's invisible in the latency budget.

**POC5 — response cache:**
- Built a read-through cache keyed by a SHA-256 of the request; forced `temperature=0` so caching is semantically correct.
- **Cache hit ~616× faster** (~4 ms vs ~2.6 s) and a repeat-heavy workload hit **62.5%, saving ~13 s of ~21 s**. Documented when caching an LLM is safe vs unsafe.

**POC6 — prefix cache:**
- Built the **radix/prefix-tree** that detects shared prompt prefixes (the RadixAttention idea) and a server that measures the real KV-cache reuse.
- **Prefill on a 438-token shared system prompt dropped from 1327 ms (cold) → ~130 ms (warm), a ~10× speedup** — the win that makes system-prompt / few-shot / agent workloads cheap.

**POC7 — KV cache observer:**
- Designed an experiment to **prove the KV cache works** without engine internals: sweep prompt length, keep output fixed, watch the fingerprint.
- A **41× longer prompt made prefill ~32× slower but decode speed stayed flat (94% kept)** — only possible if decode reuses the cache. Also surfaced prefill (compute-bound, ~600 tok/s) vs decode (memory-bound, ~30 tok/s).

**POC8 — mini-vLLM (capstone):**
- Dropped Ollama and ran a real model (distilgpt2, PyTorch + transformers) to **own the KV cache and the batching loop** ourselves.
- Manual KV-cache decode loop = **3.4× faster** than recompute; a from-scratch **continuous-batching scheduler** (admission + eviction over a shared cache) scaled throughput **1× → 4.5×** with batch size — the mechanism behind vLLM/TGI, with PagedAttention named as the next step.

---

## 🛠️ Stack

`Python` · `FastAPI` · `Uvicorn` · `Ollama` / `llama.cpp` · `Qwen2.5` — moving toward `vLLM`, `SGLang`, `TensorRT-LLM`, `Triton`, and GPU deployment in later POCs.

## ▶️ Quickstart

```bash
git clone https://github.com/sunnyRK/inference-poc-learnings.git
cd inference-poc-learnings
python3 -m venv venv
./venv/bin/pip install -r POC1-local-inference-server/requirements.txt
ollama pull qwen2.5:3b
# then follow POC1's README to run the server + benchmark
```

---

## 🎯 Goal

Deeply understand how LLMs are **served, optimized, and scaled** in production — and build a public, benchmarked portfolio that proves it. Not "I watched videos about inference," but "here's a progression from theory → working code → numbers → production architecture."

*Built in public. Follow along as the POCs land.*
