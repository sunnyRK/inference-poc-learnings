# POC6 — Prefix Cache

**What it is (in 3 lines):** When many requests share the same long beginning (a system prompt, few-shot examples), the model should process that shared **prefix only once** and reuse its KV cache for every later request — instead of re-reading it every time. We measure the real effect: prefill time on a 438-token shared prefix dropped from **1327 ms (cold) → ~130 ms (warm), a ~10× speedup**. This is what SGLang's *RadixAttention* and vLLM's *automatic prefix caching* do in production.

> 📚 Background: [04-kv-cache](../notes/04-kv-cache.md) (what the KV cache is), [01-what-is-inference](../notes/01-what-is-inference.md) (prefill vs decode). Contrast with [POC5](../POC5-response-cache) (whole-answer cache).

---

## POC5 vs POC6 (read this first)

```
POC5 Response Cache             POC6 Prefix Cache
──────────────────              ──────────────────
caches the FINAL ANSWER         caches the model's WORK on the SHARED PREFIX (KV cache)
needs the WHOLE prompt to match needs only the BEGINNING to match
hit → skip the model entirely   hit → still run the model, but do far less prefill work
lives in your server (a dict)   lives in the model engine (the KV cache)
```

Example — a chatbot with a 2000-token system prompt:
- `[system prompt] + "What is Python?"` and `[system prompt] + "What is Rust?"` are **different** full prompts → POC5 can't help (different answers).
- But they **share the first 2000 tokens** → POC6 reuses that prefix's KV cache, so the second request skips re-processing 2000 tokens. **Prefix caching helps even when the questions differ.**

---

## Why this matters in production

This is one of the biggest real-world wins, because production prompts are *mostly shared prefix*:

- **System prompts** — every request in an app starts with the same long instructions.
- **Few-shot examples** — the same examples are prepended to every call.
- **Chat history** — turn 5 of a conversation repeats turns 1–4 as its prefix.
- **Agents / RAG** — the same tool definitions or retrieved context lead every prompt.

Without prefix caching you pay to re-process that shared prefix on **every single request**. With it, you pay **once**. That directly cuts **TTFT** (the prefix is most of the prefill) and frees GPU time for real work. SGLang reported multi-× throughput gains from this on shared-prefix workloads.

---

## Two parts to this POC

### 1. `radix_tree.py` — the data structure (how engines *detect* shared prefixes)

A minimal trie over tokens. You insert prompts you've seen; for a new prompt it tells you how many leading tokens are already cached (the reusable part). Real engines (SGLang RadixAttention) use exactly this idea on real model tokens.

```
$ python radix_tree.py
Request A: You are a helpful expert programming assistant answer concisely What is Python ?
Request B: You are a helpful expert programming assistant answer concisely What is Rust ?

Shared leading tokens : 11 / 13
Reusable (cached)     : You are a helpful expert programming assistant answer concisely What is
Must compute fresh    : Rust ?
=> REUSE the KV cache for 11 tokens, only run prefill on the 2 new tokens.
```

### 2. `prefix_server.py` + `benchmark.py` — the real measured speedup

The server forwards `(system_prompt, question)` to Ollama and reads `prompt_eval_duration` (= prefill time ≈ TTFT). It tracks each prefix and reports the warm-vs-cold speedup. We don't manage the GPU KV cache ourselves — Ollama/llama.cpp reuses the shared prefix automatically; our job is the registry + measurement (the RadixAttention "brain", minus the GPU internals).

---

## Architecture

```
   {system_prompt (PREFIX), question (SUFFIX)}
                  │
                  ▼
        ┌──────── PREFIX SERVER (:8000) ────────┐
        │ prefix_id = sha256(system_prompt)      │
        │ seen this prefix before?  (registry)   │
        │ forward to Ollama, read prefill time   │
        │ report warm/cold + speedup vs cold     │
        └───────────────────┬────────────────────┘
                            ▼
                ┌────────────────────────┐
                │ Ollama / llama.cpp      │
                │ KV cache reuses the     │  ← the actual prefix reuse happens here:
                │ shared prefix automatically │  cold = full prefill, warm = skip the prefix
                └────────────────────────┘
```

---

## How to run

```bash
# 1. See the data structure
cd POC6-prefix-cache
../venv/bin/python radix_tree.py

# 2. Start the server and measure the real speedup
../venv/bin/uvicorn prefix_server:app --host 127.0.0.1 --port 8000
# (other terminal)
../venv/bin/python benchmark.py
```

---

## Results (real, measured — Apple M4)

Same 438-token system prompt (the prefix), 5 different questions (the suffix):

```
 #  state    prefill   speedup   question
 1  COLD    1327.6 ms    1.0x    What is Python?     ← first time: full prefill of the prefix
 2  WARM     130.2 ms   10.2x    What is Rust?       ← prefix reused, only the question is new
 3  WARM     184.2 ms    7.2x    What is Go?
 4  WARM     212.9 ms    6.2x    What is Java?
 5  WARM     149.7 ms    8.9x    What is C++?

different system prompt (new prefix): COLD, nothing to reuse
```

### How to read this
1. **Request 1 is COLD** — the model processes the entire 438-token prefix from scratch: **1327 ms** of prefill.
2. **Requests 2–5 are WARM** — same prefix already in the KV cache, so prefill only covers the *new question* (~3–5 tokens): **~130–210 ms**, a **6–10× speedup**.
3. **A new system prompt is cold again** — there's nothing shared to reuse. Prefix caching only helps the *shared* part.
4. **`prompt_tokens` stays 438** even when warm — Ollama still *reports* the full prompt length, but the *time* drops because the cached prefix isn't recomputed. The token count is the bill; the time is the work actually done.

**One line:** the model processes a shared prefix **once**, not once-per-request — turning a 1.3 s prefill into ~0.15 s for every repeat.

---

## Important caveat (honest engineering)

Prefix reuse depends on the prefix still being **in** the cache. Ollama here has limited cache slots, so if a request with a *different* prefix comes in between, it can **evict** your prefix and the next "warm" request pays cold again. Real engines (SGLang, vLLM) keep a large, smart cache (a radix tree with reference counting + LRU eviction) so many prefixes stay warm at once. Our benchmark sends same-prefix requests consecutively to show the clean effect; production scheduling is harder.

## Honest limitations (→ future work)
- **We measure, but don't control, the KV reuse** — it happens inside Ollama. To truly *own* prefix caching you implement the KV-cache + attention yourself (that's [POC8 — mini-vLLM]).
- **`radix_tree.py` uses words as stand-in tokens** — real engines key on actual model tokens and store pointers to GPU KV blocks.
- **No eviction policy in our registry** — it only tracks cold times; it doesn't manage cache capacity.

## Files
- `radix_tree.py` — the prefix-tree data structure + a runnable demo.
- `prefix_server.py` — server that tracks prefixes and measures warm-vs-cold prefill.
- `benchmark.py` — shared-prefix experiment showing the ~6–10× prefill speedup.
- `requirements.txt` — pinned deps.
