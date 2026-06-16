# POC3 — Streaming Inference

**What it is (in 3 lines):** Instead of waiting for the whole answer and sending it in one lump (POC1), the server sends each word *the moment the model writes it*. The total time is about the same, but the user sees the **first word almost instantly** — so it *feels* far faster. This is how ChatGPT types its reply out live.

> 📚 Simple guide to the numbers: [08-understanding-the-numbers](../notes/08-understanding-the-numbers.md). Theory: [01-what-is-inference](../notes/01-what-is-inference.md) (prefill vs decode, TTFT).

---

## The one new word: TTFT

**TTFT = Time To First Token** = how long until the user sees the **first** word.

```
NON-STREAMING (POC1):   ask ────[ blank screen, 2.9s ]────► WHOLE answer appears at once
                                                            TTFT = 2.9s  (you wait it all)

STREAMING (POC3):       ask ─[0.3s]─► word word word word word ... (words keep flowing)
                                      TTFT = 0.3s  (first word is almost instant)
```

The model is **not** faster. The same 80 tokens still take the same time to generate. Streaming only changes *when you start seeing them* — and that changes everything about how fast it **feels**.

---

## Why this matters in production

- **Perceived speed.** A 3-second blank screen feels broken. Words appearing in 0.3s feels alive. Same model, totally different user experience.
- **It's basically free.** We proved in [POC2](../POC2-concurrent-requests) that we can't easily make the GPU produce tokens faster on this hardware. Streaming improves the *felt* experience **without** needing more speed or more hardware.
- **Every real chat product streams.** OpenAI, Anthropic, Gemini all stream tokens. The format we use here (Server-Sent Events, the `data: ...` lines) is the **same one OpenAI's streaming API uses.**

---

## Architecture

```
   benchmark.py                       stream_server.py                   Ollama
 ┌───────────────┐                 ┌────────────────────┐            ┌──────────────┐
 │ reads tokens  │   GET tokens    │  /chat/stream      │  stream=   │ qwen2.5:3b   │
 │ as they       │ ◄────────────── │  StreamingResponse │  True      │ writes 1     │
 │ arrive;       │  data: {tok} \\n│  yields each token │ ◄───────── │ token at a   │
 │ records TTFT  │  data: {tok} \\n│  as an SSE event   │  one JSON  │ time         │
 │ (first token) │  data: [DONE]   │                    │  line/token│              │
 └───────────────┘                 └────────────────────┘            └──────────────┘
```

The trick is on every layer: Ollama emits one JSON line per token (`"stream": true`), the server re-emits each as an SSE event (`StreamingResponse`), and the client reads them one-by-one and stamps the time the **first** one lands.

---

## How to run

```bash
# 1. Start the streaming server
cd POC3-streaming-inference
../venv/bin/uvicorn stream_server:app --host 127.0.0.1 --port 8000

# 2. In another terminal, run the benchmark (also shows a live demo)
../venv/bin/python benchmark.py

# Or watch streaming yourself with curl (-N = don't buffer, show live):
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"message":"Explain KV cache in simple way"}'
```

---

## Results (real, measured)

**Hardware:** Apple M4, 16 GB · **Model:** Qwen2.5-3B (Q4) · `num_predict=80`.

```
prompt                              mode         TTFT     total
──────────────────────────────────────────────────────────────
Explain KV cache in simple way      non-stream   3.04 s   3.04 s
                                    streaming    0.29 s   3.49 s
What is batching in LLM inference?  non-stream   2.99 s   2.99 s
                                    streaming    0.24 s   2.86 s
Why GPU memory matters?             non-stream   2.78 s   2.78 s
                                    streaming    0.40 s   3.22 s
──────────────────────────────────────────────────────────────
avg TTFT non-streaming : 2.94 s   ← blank screen this long
avg TTFT streaming     : 0.31 s   ← first word this fast
=> first word appears ~9.5x sooner with streaming
```

### How to read this
1. **Look at the TTFT column.** Non-streaming ≈ 2.9s, streaming ≈ 0.3s. The first word shows up about **9.5× sooner**. That's the win.
2. **Look at the total column.** Both ≈ 3s. The full answer takes about the same time either way — **streaming did not make the model faster.**
3. **Streaming's total is sometimes a hair higher** (3.49s vs 3.04s) because sending one tiny network message per token adds a little overhead. That's a fine trade: you pay a few ms of total time to cut the *felt* wait by ~9×.

**The lesson in one line:** streaming doesn't change *how long* the answer takes — it changes *when you start seeing it*, and that's what users actually feel.

---

## SSE format (what's on the wire)

Each token is one event. This is the OpenAI-style format:

```
data: {"token": "Sure"}

data: {"token": "!"}

data: {"token": " Let"}

data: {"done": true, "eval_count": 80}

data: [DONE]
```

`data: ` + a JSON payload + a blank line. The token is wrapped in JSON so spaces and newlines inside a token don't break the stream.

---

## Connection to the bigger picture

- TTFT is driven by the **prefill** phase (reading your prompt) — see [01-what-is-inference](../notes/01-what-is-inference.md). Shorter prompts → smaller TTFT.
- Real engines (vLLM, TGI) stream *and* batch at the same time — they fill many users' streams from one continuously-batched GPU loop ([05-batching](../notes/05-batching.md)).
- Next optimizations attack *total* time and cost: caching repeated work ([POC5 response cache], [POC6 prefix cache]).

## Files
- `stream_server.py` — FastAPI server with `/chat` (non-stream) and `/chat/stream` (SSE).
- `benchmark.py` — measures TTFT for both modes + a live streaming demo.
- `requirements.txt` — pinned deps.
