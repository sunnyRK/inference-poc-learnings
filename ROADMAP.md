# Roadmap — 20 Inference POCs

Building inference engineering from first principles, one POC at a time.
Status: 9 done, 11 to go.

---

POC1 — Local Inference Server  [DONE]
A basic server that sends a prompt to a model and measures the speed.
Learned how a request reaches a model and how to measure latency and tokens/sec.

POC2 — Concurrent Requests  [DONE]
Hit the server with many users at the same time and measured what happens.
Found that naive serving does not scale — throughput stays flat, waiting grows.

POC3 — Streaming Inference  [DONE]
Send each word the moment it is made, instead of waiting for the whole answer.
Cut the time-to-first-word about 9x — same total time, but feels much faster.

POC4 — LLM Gateway  [DONE]
A front door adding API keys, rate limits, model routing, and usage stats.
Learned the production plumbing that wraps a model before users reach it.

POC5 — Response Cache  [DONE]
If the exact same prompt comes again, return the saved answer instantly.
About 600x faster on a repeat — and learned when caching an LLM is safe.

POC6 — Prefix Cache  [DONE]
Reuse the work for a shared prompt beginning (like a long system prompt).
About 10x faster prefill — this is the idea behind SGLang's RadixAttention.

POC7 — KV Cache Observer  [DONE]
Proved the KV cache is real by experiment: long prompt, but writing speed stays flat.
Confirmed it, and saw how prefill (reading) differs from decode (writing).

POC8 — Mini vLLM  [DONE]
Dropped Ollama and built our own engine: our own KV cache + continuous batching.
Throughput scaled up to 4.5x with batch size — the core trick behind vLLM.

---

POC9 — Benchmark Lab  [DONE]
One tool to load-test any setup and save the numbers (latency, tokens/sec, p95).
Compares configs with saved JSON + ASCII charts; re-confirmed concurrency and streaming effects.

POC10 — Quantization  [PENDING]
Run the model in smaller number formats (16-bit to 8-bit to 4-bit) to use less memory.
Learn the trade-off: smaller, faster, cheaper — but a little accuracy is lost.

POC11 — RAG Inference Server  [PENDING]
Add a search step before the model: fetch relevant text, add it to the prompt, then answer.
Learn how chatbots answer from your own documents, and how it grows the prompt.

POC12 — GPU Deployment  [PENDING]
Move from the Mac CPU to a real cloud GPU and run the same model there.
Learn GPU memory limits, cost per hour, and how the numbers change on real hardware.

POC13 — vLLM Production Server  [PENDING]
Run the real vLLM engine and re-run the POC2 load test against it.
Finally measure the 10-24x throughput win that the mini vLLM only hinted at.

POC14 — Multi-GPU Inference  [PENDING]
Split one big model across 2 or more GPUs so it fits and runs faster.
Learn tensor and pipeline parallelism — how giant models are served at all.

POC15 — Inference Dashboard  [PENDING]
A live web page showing requests, tokens/sec, latency, and GPU memory in real time.
Learn the observability side — what an on-call inference engineer watches.

POC16 — OpenAI-Compatible API  [PENDING]
Build a server that speaks the exact OpenAI API format (/v1/chat/completions).
So any existing app can point at your server with zero code changes.

POC17 — AI Agent Infrastructure  [PENDING]
Serve models that call tools and run multi-step loops, where one task means many calls.
Learn why agents stress inference differently — many short, bursty, shared-prefix calls.

POC18 — TensorRT-LLM / SGLang  [PENDING]
Try the other top engines (NVIDIA's TensorRT-LLM, and SGLang with its prefix cache).
Learn the strengths of each and when to pick which.

POC19 — Triton Kernels  [PENDING]
Write a tiny custom GPU kernel (a small math function that runs on the GPU).
Touch the lowest level — how the actual math is made fast on hardware.

POC20 — FlashAttention Deep Dive  [PENDING]
Study the trick that makes attention use far less memory by computing it in tiles.
Learn the most important speedup in modern attention — the deep end.

---

Next up: POC10 (Quantization) — runs on the laptop.
The GPU ones (POC12, POC13, POC14) need a cloud GPU when ready.
