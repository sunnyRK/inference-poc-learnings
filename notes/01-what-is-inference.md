# 01 — What Is Inference?

> Mental model for a backend/blockchain engineer moving into AI infra.

## The one-sentence definition

**Inference is running a *already-trained* model forward to produce output.**
No weights change. You feed in tokens, the model computes, you get tokens back.

- **Training** = writing to the database (expensive, batch, done rarely, updates weights).
- **Inference** = reading from the database under live traffic (latency-sensitive, runs millions of times, weights are frozen).

If you've run a blockchain node: **training is like syncing/building state; inference is like serving an `eth_call` against frozen state.** The heavy state is fixed; your job is to answer queries against it fast, cheaply, and concurrently.

---

## Why inference is the hard, valuable part

Most companies will **never train a model**. They take an open-weight model (Llama, Qwen, Mistral) or call an API. But *everyone* who ships AI has to **serve** it. That serving layer is where the money and the engineering pain live:

- A single H100 GPU costs ~$2–4/hour. If your serving is 2× inefficient, you literally pay 2× the cloud bill for the same traffic.
- Users feel latency directly. A 4-second first token kills a chat product.
- Throughput determines how many users one GPU can serve → unit economics.

> This is *exactly* the backend skillset — latency, throughput, concurrency, caching, queueing, resource utilization — applied to a new kind of workload (matrix math on GPUs).

---

## What actually happens during one inference request

A language model is **autoregressive**: it generates **one token at a time**, and each new token depends on all previous tokens.

```
Prompt: "The capital of France is"

Step 1: model reads the whole prompt  ──► predicts "Paris"
Step 2: input = prompt + "Paris"       ──► predicts "."
Step 3: input = prompt + "Paris."      ──► predicts <eos>  (stop)
```

So generating 100 output tokens = **100 forward passes** through the network. This single fact explains almost everything about inference performance.

### The two phases of every request

```
        ┌─────────────────────────┐     ┌──────────────────────────────┐
        │   PREFILL (prompt)       │     │   DECODE (generation)         │
        │                          │     │                               │
Input:  │  all prompt tokens at    │ ──► │  1 token in, 1 token out,     │
        │  once (parallel)         │     │  repeated N times (serial)    │
        └─────────────────────────┘     └──────────────────────────────┘
         Compute-bound, fast              Memory-bandwidth-bound, slow
         → drives TTFT                    → drives tokens/sec
```

1. **Prefill** — the model processes your *entire prompt in parallel* in one big matrix multiply. This is compute-heavy but fast because GPUs love big parallel matmuls. The time here ≈ **Time To First Token (TTFT)**.

2. **Decode** — the model then emits output tokens **one at a time**, each pass reusing the prompt's computation (see [[04-kv-cache]]). Each step is small and serial, so the GPU is underutilized and bottlenecked on **memory bandwidth**, not compute. This phase determines your **tokens/second**.

**Key insight:** prefill and decode have *opposite* performance characteristics. Production engines (vLLM, TensorRT-LLM) schedule them differently, and "chunked prefill" / "disaggregated prefill" exist precisely to manage this split.

---

## The vocabulary you must own

| Metric | What it means | Backend analogy | Driven by |
|---|---|---|---|
| **TTFT** (Time To First Token) | Delay before the first output token appears | Time-to-first-byte | Prompt length, queueing, prefill speed |
| **TPOT / ITL** (Time Per Output Token / Inter-Token Latency) | Gap between each streamed token | Per-row stream latency | Memory bandwidth, model size |
| **Throughput** (tokens/sec, req/sec) | Total work the server does across all users | RPS / QPS | Batching, GPU utilization |
| **Latency** (end-to-end) | TTFT + (output_tokens × TPOT) | Total request time | All of the above |
| **Goodput** | Throughput that *meets* your latency SLA | "successful RPS under SLO" | Scheduling quality |

**The central tension:** throughput vs latency. You raise throughput by **batching** more users together (see [[05-batching]]), but bigger batches can raise each user's latency. The whole game of an inference engine is maximizing throughput *without* blowing the latency SLA. That's it. That's the job.

---

## Where the time and money go

For a single GPU serving an LLM, the dominant cost in the decode phase is **moving the model's weights from GPU memory (HBM) into the compute units — for every single token.**

- A 7B model in 16-bit = ~14 GB of weights.
- To generate 1 token, you stream ~14 GB through the chip.
- At ~3 TB/s of memory bandwidth, that's a hard floor of ~5 ms/token *just to read weights* — before any actual math.

This is why decode is **memory-bandwidth-bound**, why **quantization** (smaller weights → less to move) speeds things up, and why **batching** is free-ish (read the weights once, apply to many users' tokens at once). Hold this mental model; it explains the next 15 POCs.

---

## The inference-engine landscape (where we're headed)

| Engine | Who / what | Why it matters |
|---|---|---|
| **Ollama** | llama.cpp wrapper, local-first | Easiest local serving — **our POC1** |
| **vLLM** | UC Berkeley → industry standard | Invented **PagedAttention** + continuous batching |
| **SGLang** | Fast, structured generation | **RadixAttention** = automatic prefix cache |
| **TensorRT-LLM** | NVIDIA | Fastest on NVIDIA GPUs, compiled kernels |
| **Triton Inference Server** | NVIDIA | Production serving infra (batching, multi-model) |
| **TGI** | Hugging Face | Text Generation Inference |

We will **rebuild the core ideas of these systems from scratch** (response cache, prefix cache, KV-cache visualizer, a mini-vLLM) so that by the end you don't just *use* vLLM — you understand *why* every design choice exists.

---

## What POC1 demonstrates

POC1 is the "hello world": a FastAPI server that forwards a chat request to a local model (Ollama + Qwen2.5-3B) and **measures** latency, output tokens, and tokens/sec. It's deliberately naive — no batching, no caching, no streaming, no concurrency. Every later POC removes one of those limitations and we measure the win. See [[06-poc1-learnings]].

---

### Further reading
- vLLM paper: *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., 2023)
- *Mistral / Anthropic / OpenAI* serving blog posts on TTFT vs throughput tradeoffs

Next: [[02-tokenization]] — the unit of everything we just measured.
