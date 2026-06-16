# 12 — Continuous Batching & Mini-vLLM (POC8)

> The capstone idea, in simple engineer words. This is what vLLM/TGI do, and what POC8 builds.

## The problem

We have ONE expensive GPU and MANY requests. From [[07-poc2-learnings]] we saw that serving them **one at a time** wastes the GPU — throughput stays flat, users just queue.

The fix is **batching**: run many requests **together** in one forward pass, so the cost of each step is shared. From [[05-batching]]: this is nearly free, because the expensive part (reading the model weights) is done once and applied to all requests in the batch.

## Static batching vs continuous batching

- **Static batching:** group N requests, run them together until **all** finish. Problem: requests have different lengths. A batch of one 20-token and one 500-token request must wait for the 500 — short requests finish early and their slot sits idle. (Head-of-line blocking.)

- **Continuous batching:** manage the batch **every token**. The instant a request finishes, **evict** it and **admit** a waiting one into its slot. The batch stays full; nobody waits for an unrelated long request. This is the vLLM/TGI core trick.

```
   step t:   [A][B][C][D]      ← 4 requests decoding together
   A finishes → evict A, admit E from the queue
   step t+1: [E][B][C][D]      ← batch stays full, GPU stays busy
```

## What POC8 built (our own mini engine)

We dropped Ollama and ran a real model ourselves, so OUR code owns:
1. **The KV cache** — we hold it, pass it each step, watch it grow (Part 1).
2. **The scheduler** — admit from the queue, evict when done, one shared cache for the active batch (Part 2).

## What we measured

```
mode                    tok/s     vs sequential
sequential (1 at a time)  45.6        1.0x
continuous (batch 2)      97.9        2.1x
continuous (batch 4)     153.5        3.4x
continuous (batch 8)     205.0        4.5x
```

Same model, same total work (275 tokens). Just by **batching more requests per forward pass**, throughput went up ~4.5×. We didn't make the model faster — we stopped running it one request at a time.

## The honest gap to real vLLM

When a request joins or leaves our batch, we **rebuild** the KV cache (re-process the active sequences). That's wasteful. vLLM's **PagedAttention** fixes this: it stores the KV cache in small fixed-size pages (like OS memory pages), so it can add/remove a sequence's cache without recomputing anything. That's the one big idea we *didn't* build — and now you know exactly what problem it solves.

## The one line to remember

> Continuous batching = keep the batch full every single token (evict finished, admit waiting). More requests per forward pass = more throughput, for free. PagedAttention is what makes the evict/admit cheap.

Related: [[05-batching]] · [[04-kv-cache]] · [[07-poc2-learnings]] (the flat-throughput problem this solves).
