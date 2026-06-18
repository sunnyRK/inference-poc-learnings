# POC9 — Benchmark Lab

**What it is (in 3 lines):** One reusable load-test tool for any LLM endpoint. Pick a concurrency, streaming on/off, and request count; it reports the standard serving metrics (throughput, p50/p95/p99 latency, TTFT) and **saves each run to JSON** so you can compare setups side by side. It turns the one-off scripts from POC2/POC3/POC7 into a single repeatable harness.

> 📚 Builds on [POC2](../POC2-concurrent-requests) (concurrency), [POC3](../POC3-streaming-inference) (streaming/TTFT), [08-understanding-the-numbers](../notes/08-understanding-the-numbers.md) (p50/p95/p99).

---

## Why this matters

"Is config A faster than config B?" should be answered with **numbers, saved and comparable** — not a gut feeling or a screenshot you lose. Every serious inference team has an internal benchmark harness exactly like this. It's how you justify "we switched to streaming" or "vLLM gave us 12×" with evidence. When we get to vLLM ([POC13](../ROADMAP.md)), we'll point this same tool at it and compare against today's baseline.

---

## Two scripts

- **`bench.py`** — runs one load test and saves `results/<label>.json`.
- **`compare.py`** — reads all saved results and prints a comparison table + ASCII bars.

It talks to **Ollama's `/api/chat` directly**, so it works without any POC server running. Concurrency uses the same thread-pool load generator as POC2.

---

## How to run

```bash
cd POC9-benchmark-lab

# run a few scenarios (each saves a JSON)
../venv/bin/python bench.py --label noStream-c1 --concurrency 1 --requests 8
../venv/bin/python bench.py --label noStream-c4 --concurrency 4 --requests 8
../venv/bin/python bench.py --label stream-c4   --concurrency 4 --requests 8 --streaming

# compare everything saved so far
../venv/bin/python compare.py
```

Flags: `--label` (name), `--concurrency`, `--requests`, `--max-tokens` (default 64), `--streaming`.

---

## Results (real, measured — Apple M4)

```
label            conc stream  req/s   tok/s    p50    p95    p99   TTFT
----------------------------------------------------------------------
noStream-c1         1  False   0.47    25.8  2.361  2.787  2.862  2.141
noStream-c4         4  False   0.52    28.8  7.335  7.863  7.996  6.512
stream-c4           4   True   0.50    27.9  7.714  9.094  9.144  4.802

TTFT (lower = snappier):
  noStream-c1   █████████████████ 2.14s
  noStream-c4   ████████████████████████████████████████ 6.51s
  stream-c4     ██████████████████████████████████████ 4.80s
```

### What the comparison shows (it re-confirms the whole series in one tool)
1. **Concurrency 1 → 4: throughput barely moves** (25.8 → 28.8 tok/s) but **latency triples** (p50 2.4s → 7.3s). That's [POC2](../POC2-concurrent-requests) again — on one small GPU, more users mostly means more waiting, not more work done.
2. **Streaming vs non-streaming at the same load: same throughput, but TTFT drops 6.5s → 4.8s.** That's [POC3](../POC3-streaming-inference) — streaming delivers the first token sooner even under concurrency, so it *feels* faster without changing the total.
3. **The tails (p95/p99) sit well above p50 under load** — the contention signature. Exactly why we measure percentiles, not averages ([note 08](../notes/08-understanding-the-numbers.md)).

**One line:** same harness, three configs, saved numbers — now any future change (vLLM, quantization, a new GPU) can be measured against this baseline instead of argued about.

---

## Design notes
- **Saved JSON per run** (`results/*.json`) captures config + metrics + timestamp, so results are reproducible and diffable in git.
- **Streaming TTFT** is measured as time-to-first-content-chunk; non-streaming TTFT = full latency (you get nothing until the end).
- **Token count** comes from Ollama's `eval_count` (falls back to counting streamed chunks).
- **Warmup request** runs first so model cold-start doesn't pollute the numbers.

## Limitations (→ future work)
- Targets Ollama's API only; a `--url` flag + OpenAI-format support would let it benchmark vLLM/TGI/OpenAI directly (easy add for [POC13](../ROADMAP.md)).
- Fixed prompt set; real load tests replay production traffic shapes.
- No latency-vs-throughput sweep plot yet — `compare.py` is text/ASCII only.

## Files
- `bench.py` — the load-test runner (saves results/<label>.json).
- `compare.py` — table + ASCII chart across all saved runs.
- `results/*.json` — saved benchmark runs (committed as portfolio artifacts).
