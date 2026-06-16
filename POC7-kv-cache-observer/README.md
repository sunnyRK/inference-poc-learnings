# POC7 — KV Cache Observer & Verifier

**What it is (in 3 lines):** You can't read Ollama's raw KV-cache blocks (only vLLM/TensorRT-LLM expose those). So instead we **prove the KV cache is working by experiment** — sweep the prompt length and watch the fingerprint: prefill time grows a lot, but decode speed stays flat. We measured a 41× longer prompt make prefill ~32× slower while decode speed barely moved (kept 94%) → **KV cache confirmed.**

> 📚 Background: [04-kv-cache](../notes/04-kv-cache.md), [01-what-is-inference](../notes/01-what-is-inference.md) (prefill vs decode), [10-glossary](../notes/10-glossary.md).

---

## The question this POC answers

We *talk* about the KV cache a lot ([note 04](../notes/04-kv-cache.md)), and in [POC6](../POC6-prefix-cache) we saw shared-prefix reuse. But how do you **prove** the KV cache is even there, when Ollama never returns a `kv_cache_used: true` field? Answer: design an experiment whose result is only possible *if* the cache exists.

## The fingerprint of a working KV cache

Recall the two phases ([note 01](../notes/01-what-is-inference.md)):
- **Prefill** processes the whole prompt once and builds the KV cache.
- **Decode** generates output tokens one at a time, *reusing* that cache.

So if the KV cache is real:
- **Prefill time should grow** as the prompt gets longer (more tokens to process once).
- **Decode speed (tok/s) should stay ~flat** regardless of prompt length — because each new token reuses the stored K/V instead of re-reading the whole prompt.

**The key logic:** if there were **no** KV cache, decode would re-process the entire context for *every single output token*, so decode tok/s would **collapse** on long prompts. It doesn't → the cache is real.

```
   NO KV cache (hypothetical)        WITH KV cache (reality)
   decode slows down as prompt        decode speed stays flat;
   grows (re-reads everything)        only prefill grows
   tok/s ●                            tok/s ●──●──●──●──●  (flat)
         │●                                 prefill ●
         │ ●                                       ●
         │  ●●●                            ●──●
         └─────── prompt length            └─────── prompt length
```

## What you measure (Ollama's metrics — durations are in NANOSECONDS)

| Metric | Meaning | Phase |
|---|---|---|
| `prompt_eval_count` | input/prompt tokens processed | prefill |
| `prompt_eval_duration` | time to process the prompt | **prefill time** |
| `eval_count` | output tokens generated | decode |
| `eval_duration` | time to generate them | **decode time** |
| `load_duration` | model load time (cold start) | — |

`prefill_tok/s = prompt_eval_count / prompt_eval_duration` · `decode_tok/s = eval_count / eval_duration`.

---

## How to run

```bash
cd POC7-kv-cache-observer
../venv/bin/python kv_observer.py     # talks straight to Ollama on :11434, no server needed
```

---

## Results (real, measured — Apple M4)

Prompt length swept from tiny to large; **output fixed at 80 tokens** every time:

```
 in_tok | prefill_s | prefill_tps || out_tok | decode_s | decode_tps
--------+-----------+-------------++---------+----------+-----------
     42 |     0.085 |       494.1 ||      80 |    2.407 |      33.23
    211 |     0.636 |       331.7 ||      80 |    2.680 |      29.85
    463 |     0.848 |       545.8 ||      80 |    2.679 |      29.86
    883 |     1.338 |       659.7 ||      80 |    2.526 |      31.67
   1723 |     2.717 |       634.1 ||      80 |    2.559 |      31.27

VERDICT
  prompt grew         : 42 -> 1723 tokens (~41x)
  prefill time grew   : 0.085s -> 2.717s (~32x)
  decode speed change : 33.2 -> 31.3 tok/s (94% of the small-prompt speed)
  ==> KV CACHE CONFIRMED. Prefill builds it once; decode reuses it.
```

### How to read this — three lessons

1. **Prefill scales with prompt length.** 41× more input tokens → ~32× more prefill time. Long prompts are expensive *once*, up front. This is your **TTFT** cost.

2. **Decode speed is flat (~30–33 tok/s) no matter the prompt length.** A 1723-token prompt decodes at basically the same speed as a 42-token one (94% retained). **This is the proof:** the model is *not* re-reading the prompt for each token — it reuses the KV cache. Without it, the last row's decode would be ~40× slower.

3. **Prefill tok/s (300–660) ≫ decode tok/s (~30).** Prefill processes tokens ~15–20× faster per token than decode — because prefill does all prompt tokens **in parallel** (compute-bound), while decode does them **one at a time** (memory-bandwidth-bound). Same model, two completely different speed regimes. (See [note 10 — compute-bound vs memory-bound](../notes/10-glossary.md).)

---

## What an inference engineer actually controls in Ollama

You can't manage raw KV blocks here, but you **can** manage everything that creates KV-cache pressure:

| Lever | Knob | Effect |
|---|---|---|
| **Context length** | `num_ctx` | bigger context = more KV memory (VRAM) per request |
| **Concurrency** | `OLLAMA_NUM_PARALLEL` | memory ≈ `num_parallel × context_length` (this was the [POC2](../POC2-concurrent-requests) ceiling) |
| **Model lifetime** | `keep_alive` (e.g. `30m`, `-1`, `0`) | keeps the model resident → avoids cold-start reload (NOT the same as KV cache) |
| **History size** | trim old turns | shorter prompt → cheaper prefill, less KV memory |

> `keep_alive` keeps the **model weights** loaded; it is *not* the KV cache. The KV cache is per-request attention memory built during prefill. Don't confuse "model stays warm" with "KV cache reused."

## The honest boundary (what ChatGPT was right about)
From these metrics you **infer** the KV cache; Ollama doesn't expose a `kv_cache_used` flag. The experiment above is how you turn an inference into proof. To *directly* allocate, write, page, and evict KV blocks, you need a real engine — that's [POC8 — mini-vLLM], where you implement the cache yourself.

## Files
- `kv_observer.py` — sweeps prompt length, measures prefill vs decode, prints the verdict.
