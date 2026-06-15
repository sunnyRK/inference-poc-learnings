# 03 — Attention

> Attention is the operation that makes transformers work — and the operation that makes inference expensive. Understanding it is the key that unlocks KV cache, batching, and every memory optimization that follows.

## The problem attention solves

To predict the next token, the model must look at the *relevant* earlier tokens, not just the most recent one.

```
"The trophy didn't fit in the suitcase because it was too big."
                                              └─ "it" = the trophy
```

To resolve "it", the model must **attend** back to "trophy". Attention is the mechanism that lets every token pull in information from every other token, weighted by relevance.

> Backend analogy: attention is a **soft, content-addressed lookup**. Each token issues a *query* ("what am I looking for?"), every token advertises a *key* ("what do I offer?"), and carries a *value* ("here's my actual content"). The query is matched against all keys (a dot-product = relevance score), scores are normalized (softmax → weights summing to 1), and the output is the weighted sum of values. It's a **differentiable key-value store** queried by similarity.

---

## Q, K, V — the three projections

From each token's embedding vector, the model computes three vectors via learned weight matrices:

```
token embedding x  ──►  Q = x·Wq   (Query:  what I'm looking for)
                        K = x·Wk   (Key:    what I contain, for matching)
                        V = x·Wv   (Value:  what I pass on if matched)
```

Then for one query token attending over all tokens:

```
                   Q · Kᵀ
attention = softmax(──────) · V
                    √d_k
```

- `Q · Kᵀ` → a score of how much this token cares about each other token.
- `÷ √d_k` → scaling to keep gradients/softmax stable.
- `softmax` → turn scores into weights that sum to 1.
- `· V` → blend the values by those weights = the output.

**Multi-head attention** just does this several times in parallel ("heads"), each head learning a different relationship (syntax, coreference, position…), then concatenates the results.

---

## Causal (masked) attention — why generation is one-directional

In a generative LLM, token *i* may only attend to tokens *≤ i* (the past), never the future. This is enforced with a **causal mask** that sets future scores to −∞ before softmax.

```
        attends to →
        t1   t2   t3   t4
  t1 [  ✓    ·    ·    ·  ]
  t2 [  ✓    ✓    ·    ·  ]     ✓ = allowed   · = masked (−∞)
  t3 [  ✓    ✓    ✓    ·  ]
  t4 [  ✓    ✓    ✓    ✓  ]
```

This lower-triangular structure is **the entire reason KV cache works** (next note): when you generate token 5, tokens 1–4's keys and values are *exactly the same as they were before* — the mask guarantees the past never depends on the future. So you can cache them. Hold that thought.

---

## Why attention is the cost center of inference

Attention's compute and memory scale with **sequence length `n`**:

```
Q·Kᵀ  produces an  n × n  score matrix.

cost ∝ n²   (every token scores against every other token)
```

This **quadratic** scaling is the villain of long-context inference:

- Double the context → 4× the attention compute and the score-matrix memory.
- A 100k-token context is brutal precisely because of this `n²` term.

Two complementary attacks on this cost, both of which we'll touch in later POCs:

1. **KV cache** ([[04-kv-cache]]) — removes *redundant recomputation* across decode steps. Turns per-step cost from `O(n²)` into `O(n)`. **This is the single most important inference optimization.**

2. **FlashAttention** — removes the *memory bottleneck*. Instead of writing the full `n×n` matrix to slow GPU memory (HBM), it computes attention in tiles inside fast on-chip SRAM, never materializing the whole matrix. Same math, far less memory traffic → big speedups. (A late-stage POC.)

There are also *architectural* fixes that change K/V sharing to shrink the cache:

- **MHA** (Multi-Head): every head has its own K,V. Biggest cache.
- **MQA** (Multi-Query): all heads share *one* K,V. Tiny cache, slight quality loss.
- **GQA** (Grouped-Query): groups of heads share K,V — the modern default (Llama 2/3, Qwen). The sweet spot. **This is a KV-cache-size optimization baked into the model architecture.**

---

## The mental model to carry forward

```
PREFILL:  compute Q,K,V for ALL prompt tokens at once  → big parallel matmul (GPU-friendly)
          └─► store every token's K,V in the KV cache

DECODE:   for the new token, compute its Q
          attend over ALL cached K,V (no recompute!)
          produce 1 token, append its K,V to the cache
          repeat
```

Everything in inference serving is, in some sense, **memory management for the K and V vectors** produced by attention. vLLM's PagedAttention, SGLang's RadixAttention, prefix caching, quantized KV cache — all of it is about storing, sharing, and moving these K/V tensors efficiently.

---

### Why this matters for the portfolio
When an interviewer asks "why is long-context expensive?" or "what does PagedAttention page?", the answer is *this note*: attention is `O(n²)`, the K/V tensors are the thing being cached and paged, and GQA/MQA exist to shrink them.

Next: [[04-kv-cache]] — the optimization that makes generation practical.
