# 10 — Inference Glossary (for a backend engineer)

> Plain-English definitions of the inference words, mapped onto backend/systems ideas you already know. No fluff.

---

## A. The basics

**Inference** — running an *already-trained* model to produce output. The weights never change (read-only). Like serving read queries against a fixed, frozen database. (Training = building the DB; inference = querying it under live traffic.)

**Token** — a small chunk of text, usually a piece of a word. The model never sees characters; it sees a list of token IDs (integers). Think of it as the model's serialization unit / wire format.

**Tokenization** — converting text → tokens (and back). A tokenizer does this *before* the model runs. "encode text" = text → token IDs; "decode text" = token IDs → text. (Different from the model's *decode phase* below — same English word, different meaning.)

**Autoregressive** — the model generates **one token at a time**, and each new token depends on all previous ones. The output token is appended to the input, then the model runs again. It's a `while` loop that grows its own input each iteration.

---

## B. The two phases of a request (most important section)

Every request runs in two phases with **opposite** performance characteristics:

**Prefill** — phase 1: the model reads your **whole prompt at once** in one parallel pass and builds its internal state (fills the KV cache). Compute-heavy but fast. This is what **TTFT** mostly waits on. (Like loading/warming the working set before the generation loop starts.)

**Decode** — phase 2: the model generates output tokens **one by one**, one forward pass per token. Slow and serial. This sets your **tokens/sec**. (Like the per-iteration body of the generation loop.)

```
[ prefill: read whole prompt ]  [decode t1][decode t2]...[decode t80]
       fast, parallel           └────── slow, one-at-a-time ──────┘
       → drives TTFT                    → drives tokens/sec
```

**Encode / Encoder** — careful, two meanings:
- *"encoding"* loosely = turning input into internal number vectors (what prefill does to your prompt).
- *"Encoder"* (a model type) = reads text to *understand* it but doesn't generate (e.g. BERT, embedding models).
- Our models (Qwen, Llama, GPT) are **decoder-only** — no separate encoder. They just do **prefill + decode**. So in these POCs you care about prefill and decode, not an "encoder."

---

## C. The speed words

**Latency** — total time for **one** request (TTFT + all decode steps). One user's wait. Lower = better.

**TTFT (Time To First Token)** — time from sending the request until the **first** output token appears. Mostly = prefill time. This is the LLM version of "time to first byte."

**TPOT / ITL (Time Per Output Token / Inter-Token Latency)** — the gap between each streamed token after the first. Sets how fast text *flows* on screen. Roughly `1 / TPOT = tokens per second`.

**Throughput** — total work the **whole server** finishes per second, across all users (requests/sec or tokens/sec). Server capacity. Higher = better. (Key difference: *latency* = one request's time; *throughput* = total rate. A server can have great throughput but bad latency.)

**Tokens/sec (tok/s)** — how many tokens are produced per second. Per single request = the decode speed. Across all requests = server throughput.

**Goodput** — the throughput that actually meets your latency target (SLA). "Useful" throughput. Requests served too slowly to count don't add to goodput.

---

## D. The memory words

**Parameters / weights** — the learned numbers inside the model. "3B" = 3 billion of them. They get read on **every** token. Memory size ≈ params × bytes-per-number (e.g. 3B × 2 bytes = ~6 GB in 16-bit). This is the fixed state you must load before serving.

**KV cache** — saved Key/Value vectors for every past token, so the model doesn't recompute the prompt/history on every step. Turns generation from O(n²) work into O(n). It's memoization for the generation loop. It's **big and per-request**, so it's the main thing eating GPU memory.

**Context window** — the max number of tokens (prompt + output combined) the model can handle at once, e.g. 32k. A fixed-size buffer for the whole conversation.

**Attention** — the operation where each token looks at the other tokens to decide what's relevant. Its cost grows with the **square** of the sequence length (O(n²)) — double the context, 4× the attention cost. The expensive core of a transformer.

**GPU memory (VRAM / HBM)** — the fast memory on the GPU chip that holds the weights + KV cache. Usually the **real bottleneck** (caps how many users and how long a context you can serve). HBM = "High Bandwidth Memory."

**Memory bandwidth** — how fast data moves between GPU memory and the compute units (e.g. TB/s). The decode-phase bottleneck — most decode time is spent *reading the weights*, not doing math.

**Compute-bound vs memory-bound** — compute-bound = limited by math speed (that's **prefill**). Memory-bound = limited by moving data (that's **decode**). This one distinction explains why batching is nearly free and why quantization speeds things up.

**Quantization** — storing weights/KV in a smaller number format (16-bit → 8-bit → 4-bit). Less memory used and less data to move → faster and cheaper, with a small accuracy loss. Like choosing `int8` instead of `float32`. Our `qwen2.5:3b` "Q4" model is 4-bit quantized.

**Cold start** — the extra delay on the **first** request because the weights must be loaded from disk into memory. Warm requests skip it. It's a cold cache on service boot.

---

## E. The scaling words

**Batch / Batching** — processing several requests **together** in one GPU pass, so the expensive cost of loading the weights is shared across all of them. The #1 throughput lever. Like request coalescing or batched DB writes.

**Continuous batching (a.k.a. in-flight batching)** — rebuild the batch **every token**: finished requests drop out, waiting ones join in, with no waiting for the slowest. The core trick in vLLM/TGI. A rolling worker pool that swaps tasks every tick. (See [[05-batching]].)

**Concurrency** — how many requests are being handled at the same time. Note: concurrency ≠ throughput. You can have 8 concurrent requests but flat throughput if they're all just queued and waiting (that was the [[07-poc2-learnings]] result).

**Head-of-line blocking** — when one slow request holds up others stuck behind it in the same batch/queue. The bug continuous batching fixes.

---

## F. Serving / transport words

**Streaming** — sending each output token to the client as it's produced (over chunked HTTP / SSE) instead of waiting for the whole answer. Lowers **perceived** latency (TTFT), not total time. (See [[09-streaming-and-ttft]].)

**SSE (Server-Sent Events)** — a simple one-way HTTP streaming format: a long-lived response made of `data: ...` lines separated by blank lines. What OpenAI's streaming API uses.

**p50 / p95 / p99 (percentiles)** — sort all the request latencies small→big. p50 = the middle value (the normal experience). p95 / p99 = near the slow end (the unlucky users, the "tail"). We use these instead of the average because a few very slow requests skew the average. (See [[08-understanding-the-numbers]].)

---

## The 6 you'll say most often

| Word | One line |
|------|----------|
| **Prefill** | read the whole prompt in one fast pass → sets TTFT |
| **Decode** | generate tokens one-by-one → sets tokens/sec |
| **TTFT** | time until the first token appears |
| **KV cache** | saved past tokens so you don't recompute → O(n²)→O(n), eats memory |
| **Batching** | run many requests in one GPU pass → throughput |
| **Throughput vs latency** | total rate vs one request's time |

Related notes: [[01-what-is-inference]] · [[04-kv-cache]] · [[05-batching]] · [[08-understanding-the-numbers]] · [[09-streaming-and-ttft]]
