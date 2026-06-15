# 04 — KV Cache

> If you remember one optimization from this whole journey, make it this one. The KV cache is *the* reason LLM generation is affordable, and its management is what 80% of an inference engine's code is about.

## The problem: naive generation recomputes everything

Recall: generation is autoregressive — one forward pass per output token. The naive way:

```
Step 1: forward("The capital of France is")              → "Paris"
Step 2: forward("The capital of France is Paris")        → ","
Step 3: forward("The capital of France is Paris,")       → " a"
...
```

Notice the waste: at step 2 you reprocess "The capital of France is" **again**, even though you already did that exact work at step 1. At step 100 you'd reprocess 99 tokens you've already seen. Total work to generate `n` tokens this way is `1 + 2 + 3 + ... + n = O(n²)`. Brutal.

> Backend analogy: it's like recomputing an entire aggregation from the first row every time a new row arrives, instead of keeping a running total. The fix is the same: **cache the intermediate state.**

---

## The insight: the past doesn't change

From [[03-attention]]: with the **causal mask**, token *i*'s Key and Value vectors depend only on tokens `≤ i`. When you append a new token, **none of the previous tokens' K/V vectors change.** They are immutable history.

So: compute each token's K and V **once**, stash them, and reuse them on every future step. That's the KV cache.

```
                       ┌──────────── KV CACHE (grows by 1 each step) ───────────┐
Step 1 (prefill):  K,V for [The][capital][of][France][is]   ← computed once
Step 2 (decode):   reuse cache; compute K,V for [Paris] only; append
Step 3 (decode):   reuse cache; compute K,V for [,] only;    append
```

Now each decode step only computes Q/K/V for **the single new token** and attends over the cached K/V. Per-step cost drops from `O(n)` recompute to `O(1)` new work (plus an `O(n)` attention read). Total generation: **`O(n²)` → `O(n)`**. This is the difference between "unusable" and "ChatGPT".

---

## What exactly is cached

For **every layer**, **every attention head**, **every token so far**, two vectors: K and V. The cache size:

```
kv_bytes = 2 (K and V)
         × num_layers
         × num_kv_heads          ← GQA/MQA shrink this (see [[03-attention]])
         × head_dim
         × seq_len               ← grows with every token
         × dtype_bytes           ← 2 for fp16, 1 for fp8/int8 (quantized KV)
         × batch_size            ← every concurrent request needs its own
```

**Worked example — Llama-2-7B, fp16, one 4096-token sequence:**

```
2 × 32 layers × 32 heads × 128 head_dim × 4096 tokens × 2 bytes ≈ 2.1 GB
```

**~2 GB of KV cache for a single request** — on top of the 14 GB of model weights. And it scales linearly with both sequence length *and* number of concurrent requests. This is why:

- **GPU memory, not compute, is usually the limit on how many users you can serve.**
- Long contexts are expensive in *memory*, not just compute.
- **Quantizing the KV cache** (fp16 → fp8) is a real lever — it halves this number.
- **GQA** (fewer KV heads) is why modern models can serve long contexts at all.

---

## Why this single fact reshapes inference systems

The KV cache is **large, per-request, and grows unpredictably** (you don't know how long the output will be). That creates two hard systems problems that the famous engines exist to solve:

### 1. Memory fragmentation → PagedAttention (vLLM)
If you pre-allocate a contiguous slab of "max possible length" KV memory per request, you waste enormous amounts (most requests are short) and fragment the rest. **vLLM's PagedAttention** borrows the OS virtual-memory trick: split the KV cache into fixed-size **blocks (pages)**, allocate them on demand, and keep a block table mapping logical positions → physical blocks. Result: near-zero waste, far higher concurrency. **This one idea is why vLLM beat everything in 2023.** (We build a toy version in POC8 — mini-vLLM.)

### 2. Redundant prefixes → prefix caching (SGLang's RadixAttention)
If 1,000 requests all start with the same 2,000-token system prompt, naively you compute and store that prefix's KV cache 1,000 times. **Prefix caching** stores it **once** and shares it across requests (a radix tree keyed by token prefix). Massive TTFT and memory savings for chatbots, few-shot prompts, and agents. (We build this in POC6.)

---

## How POC1 relates

POC1 uses Ollama, which (via llama.cpp) **already manages a KV cache internally** — you're benefiting from it without seeing it. The proof is in the numbers: POC1 sustains ~30–34 tokens/sec on an M4. Without a KV cache, per-token cost would *grow* as the answer got longer and tokens/sec would visibly decay across the response. It doesn't — because the cache makes each decode step roughly constant-cost.

What POC1 **doesn't** show: the memory pressure. With one request and short outputs, the cache is tiny. POC2 (concurrency) starts to expose it — each concurrent request needs its *own* KV cache, and that's where a single machine's limits appear. POC7 will **visualize** the cache directly.

---

## The interview-ready summary

- **What:** cache the per-token Key/Value vectors so you never recompute the prompt/history.
- **Why it works:** causal masking makes past K/V immutable.
- **Impact:** generation goes from `O(n²)` to `O(n)` compute.
- **The catch:** the cache is huge (GBs), per-request, and grows with length → **memory is the bottleneck**.
- **The systems response:** PagedAttention (no fragmentation), prefix caching (no duplication), KV quantization + GQA (smaller cache).

> "Inference serving is, mostly, the art of managing the KV cache." Everything from here builds on this sentence.

Next: [[05-batching]] — how to serve many of these cached requests at once.
