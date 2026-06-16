# 11 — Proving the KV Cache Is Real (POC7)

> How to turn "the KV cache exists" from a claim into a measured fact. Simple version.

## The problem

We keep saying the model has a KV cache ([[04-kv-cache]]). But Ollama never tells us `kv_cache_used: true`. So how do we *prove* it's there?

Answer: run an experiment that can only come out one way *if* the cache exists.

## The trick

Two phases per request ([[01-what-is-inference]]):
- **Prefill** = read the whole prompt once, build the KV cache.
- **Decode** = generate output tokens one by one, reusing that cache.

So here's the test. Keep the **output length fixed** (80 tokens), and make the **prompt longer and longer**. Then watch two numbers:

- **Prefill time** — should get bigger as the prompt grows (more to read once).
- **Decode speed (tokens/sec)** — should stay about the **same**, no matter how long the prompt is.

**Why decode speed is the proof:** if there were *no* KV cache, the model would re-read the entire prompt to make *each* output token. So a long prompt would make decode crawl. If decode speed stays flat even with a huge prompt, the model is clearly **not** re-reading the prompt — it's reusing the saved KV cache.

## What we measured (real numbers)

```
input tokens   prefill time     decode speed
     42          0.085 s          33.2 tok/s
   1723          2.717 s          31.3 tok/s
```

- Prompt got **~41× longer**.
- Prefill time got **~32× bigger**  ✅ (grows with prompt — expected).
- Decode speed stayed **~the same** (33 → 31, kept 94%) ✅ (flat — the proof!).

If there were no KV cache, that last decode speed would have been roughly **40× slower**. It wasn't. **So the KV cache is real.**

## Bonus thing we noticed

- Prefill ran at ~300–660 tokens/sec, but decode only ~30 tokens/sec.
- Prefill is ~15–20× faster per token because it processes the whole prompt **in parallel** (compute-bound), while decode makes tokens **one at a time** (memory-bound).
- Same model, two very different speeds — that's the prefill-vs-decode split in one measurement.

## The one line to remember

> Make the prompt huge, keep the output fixed. If decode speed stays flat, the KV cache is doing its job.

Related: [[04-kv-cache]] · [[01-what-is-inference]] · [[10-glossary]] (compute-bound vs memory-bound).
