# Inference Engineering — POC Learnings

> Building up LLM **inference & serving** infrastructure from first principles — one focused proof-of-concept at a time. Theory → implementation → benchmarks → production concepts.

I'm a senior backend & blockchain engineer moving into **AI infrastructure**. Rather than only consuming AI APIs, this repo is me learning the hard parts of *serving* LLMs in production — latency, throughput, batching, KV cache, GPU memory, distributed inference — by **building and benchmarking** each idea, then connecting it back to how real engines (vLLM, SGLang, TensorRT-LLM, Triton) work.

Every POC ships with: **runnable code · README · architecture diagram · real benchmarks · written learnings · tradeoffs · what's next.**

---

## 📊 Progress

| # | POC | Concept | Status |
|---|-----|---------|--------|
| 1 | [Local Inference Server](./POC1-local-inference-server) | Serving basics, instrumentation, latency/throughput measurement | ✅ Done |
| 2 | Concurrent Requests | Throughput, concurrency, saturation, tail latency | 🔜 Next |
| 3 | Streaming Inference | TTFT, token streaming, perceived latency | ⬜ Planned |
| 4 | LLM Gateway | Routing, rate limiting, multi-model serving | ⬜ Planned |
| 5 | Response Cache | Exact-match caching, cost reduction | ⬜ Planned |
| 6 | Prefix Cache | KV-cache reuse, RadixAttention idea | ⬜ Planned |
| 7 | KV Cache Visualizer | Making GPU memory visible | ⬜ Planned |
| 8 | Mini vLLM | Continuous batching + paged KV from scratch | ⬜ Planned |
| 9 | Benchmark Lab | Standardized load testing across engines | ⬜ Planned |
| + | Quantization · RAG serving · GPU deploy · vLLM prod · Multi-GPU · TensorRT/SGLang · Triton kernels · FlashAttention | Advanced infra | ⬜ Backlog |

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
| [07 — POC2 Plan](./notes/07-poc2-learnings.md) | Hypotheses for the concurrency benchmark (pre-registered) |

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
└── (POC2, POC3, ... added as built)
```

Each `POCx-*/` folder is self-contained: its own README, code, deps, and benchmark.

---

## 🔑 Highlights so far (from POC1)

- Built an **instrumented** FastAPI inference server over Ollama/Qwen2.5-3B that reports latency, prompt/output token counts, and tokens/sec on every request.
- Measured on Apple M4: **~32 tok/s decode ceiling**, **~3s cold-start tax**, and confirmed **latency scales linearly with output tokens** while tokens/sec stays flat (the KV cache working underneath).
- Decomposed warm latency into **~90% decode + ~10% serving overhead**, establishing the baseline that batching/streaming/caching will improve.

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
