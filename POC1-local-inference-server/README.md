# POC1 — Local Inference Server

A minimal, **instrumented** LLM inference server: FastAPI in front of a locally-served model (Ollama + Qwen2.5-3B). It exposes a `/chat` endpoint and returns not just the answer but the **performance metrics** of every request — latency, token counts, and tokens/sec.

This is the "hello world" of inference serving and the **measurement baseline** for the entire POC series.

> 📚 Theory behind this POC lives in [`../notes/`](../notes): [what-is-inference](../notes/01-what-is-inference.md) · [tokenization](../notes/02-tokenization.md) · [attention](../notes/03-attention.md) · [kv-cache](../notes/04-kv-cache.md) · [batching](../notes/05-batching.md). **Learnings + real numbers:** [06-poc1-learnings](../notes/06-poc1-learnings.md).

---

## Why this POC matters

You can't optimize what you can't measure. Before touching batching, caching, or streaming, you need a baseline that reports the metrics those optimizations will move. POC1 establishes that baseline and proves out the request path:

```
client → HTTP → FastAPI → HTTP → Ollama → llama.cpp → Metal/GPU → tokens back
```

Every later POC removes one limitation of this server (no concurrency, no streaming, no caching) and measures the win against these numbers.

---

## Architecture

```
   ┌────────────┐   POST /chat        ┌─────────────────────┐
   │  client /  │  {"message": ...}   │   FastAPI server     │
   │ benchmark  │ ──────────────────► │   (main.py)          │
   └────────────┘                     │                      │
        ▲                             │  - start timer       │
        │   JSON: answer +            │  - build chat payload│
        │   latency, tokens, tok/s    │  - POST to Ollama    │
        └──────────────────────────── │  - compute metrics   │
                                      └──────────┬───────────┘
                                                 │ POST /api/chat
                                                 ▼
                                      ┌─────────────────────┐
                                      │  Ollama (:11434)     │
                                      │  qwen2.5:3b          │
                                      │  llama.cpp + Metal   │
                                      │  (KV cache lives here)│
                                      └─────────────────────┘
```

The server is a thin, measured proxy. The real inference engine is Ollama/llama.cpp; FastAPI's job is to expose it cleanly and **instrument** it.

---

## API

### `GET /` — health check
```json
{ "status": "ok", "service": "mini-inference-server" }
```

### `POST /chat`
**Request:**
```json
{ "message": "Explain KV cache in simple way" }
```
**Response:**
```json
{
  "model": "qwen2.5:3b",
  "response": "...",
  "latency_seconds": 2.772,
  "prompt_eval_count": 38,     // input tokens  (prefill work)
  "eval_count": 80,            // output tokens (decode work)
  "tokens_per_second": 31.93   // pure decode speed = eval_count / eval_duration
}
```

`num_predict` is capped at **80** tokens — a hard ceiling on generation, which bounds latency and cost (this is the local equivalent of `max_tokens`).

---

## How to run

**Prereqs:** [Ollama](https://ollama.com) installed and the model pulled.

```bash
# 1. Pull the model (once)
ollama pull qwen2.5:3b

# 2. Make sure Ollama is serving (usually automatic)
ollama serve            # or it's already running as a background service

# 3. From the repo root, set up the venv (once)
python3 -m venv venv
./venv/bin/pip install -r POC1-local-inference-server/requirements.txt

# 4. Start the inference server
cd POC1-local-inference-server
../venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# 5. In another terminal, hit it
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Explain KV cache in simple way"}' | python3 -m json.tool

# 6. Or run the benchmark suite
../venv/bin/python benchmark.py
```

---

## Benchmark results (real, measured)

**Hardware:** Apple M4, 16 GB · **Model:** Qwen2.5-3B (Q4) · **Engine:** Ollama (llama.cpp + Metal) · sequential.

```
Prompt                              Latency   Out tokens   Tokens/sec
─────────────────────────────────────────────────────────────────────
Explain KV cache in simple way      5.673 s      80          32.87   ← cold start
What is batching in LLM inference?  2.772 s      80          31.93
Explain tokenization in 2 lines     1.265 s      33          34.21
What is time to first token?        2.880 s      80          30.50
Why GPU memory matters?             2.674 s      80          33.35
─────────────────────────────────────────────────────────────────────
Avg ≈ 32.6 tok/s  ·  warm 80-token latency ≈ 2.78 s  ·  cold-start tax ≈ 3 s
```

**Three things this baseline proves** (full analysis in [06-poc1-learnings](../notes/06-poc1-learnings.md)):
1. **Cold start is ~3s** — the first request loads 1.9 GB of weights into memory; warm requests don't pay it.
2. **tokens/sec ≈ constant (~32)** — that's the M4's decode ceiling for a 3B model. The KV cache (managed inside Ollama) keeps it flat across the whole response.
3. **Latency ∝ output tokens** — 33 tokens → 1.27s, 80 tokens → ~2.7s. Output length is the latency dial; decode is ~90% of warm latency.

---

## Limitations (intentional — each becomes a future POC)

| Limitation | Fixed by | Concept |
|---|---|---|
| One request at a time | POC2 | concurrency / throughput |
| Waits for all tokens before responding | POC3 | streaming / TTFT |
| Recomputes identical prompts | POC5 | response cache |
| Recomputes shared prefixes | POC6 | prefix cache / KV reuse |
| Relies on Ollama's scheduler | POC8 | continuous batching (mini-vLLM) |

---

## Files
- `main.py` — FastAPI server with the instrumented `/chat` endpoint.
- `benchmark.py` — sequential benchmark over 5 prompts, prints latency/tokens/tok-s.
- `requirements.txt` — pinned deps.
