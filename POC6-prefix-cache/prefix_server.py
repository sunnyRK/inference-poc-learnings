"""
POC6 — Prefix Cache Server

Takes a (system_prompt, question) pair. The system_prompt is the reusable
PREFIX; the question is the new SUFFIX. When many requests share the same
system_prompt, the model only has to process that long prefix ONCE — every
later request reuses its KV cache and prefill becomes much faster.

We don't manage the KV cache ourselves (that lives inside Ollama/llama.cpp,
which reuses the shared prefix automatically). What we add is the application
layer: a registry that tracks each prefix, records its COLD prefill time, and
reports the speedup on warm (repeat-prefix) requests. This is the same job
SGLang's RadixAttention does — minus the GPU internals.
"""

import hashlib
import threading
import time

import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"


class ChatRequest(BaseModel):
    system_prompt: str  # the reusable prefix
    question: str       # the new suffix


_lock = threading.Lock()
# prefix_id -> {cold_prefill_ms, hits, prompt_tokens, preview}
_prefixes: dict[str, dict] = {}


def prefix_id(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.strip().encode()).hexdigest()[:8]


@app.get("/")
def health():
    return {"status": "ok", "service": "prefix-cache-server"}


@app.post("/chat")
def chat(req: ChatRequest):
    pid = prefix_id(req.system_prompt)
    with _lock:
        known = pid in _prefixes

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.question},
        ],
        "stream": False,
        "options": {"num_predict": 20, "temperature": 0},  # short output; we care about prefill
    }
    t = time.time()
    data = requests.post(OLLAMA_URL, json=payload).json()
    total = time.time() - t

    # prompt_eval_duration = time spent on PREFILL (processing the prompt) ~= TTFT
    prefill_ms = round((data.get("prompt_eval_duration") or 0) / 1e6, 1)
    prompt_tokens = data.get("prompt_eval_count")

    with _lock:
        if not known:
            _prefixes[pid] = {
                "cold_prefill_ms": prefill_ms,
                "hits": 0,
                "prompt_tokens": prompt_tokens,
                "preview": req.system_prompt[:40].replace("\n", " "),
            }
        else:
            _prefixes[pid]["hits"] += 1
        cold = _prefixes[pid]["cold_prefill_ms"]

    speedup = round(cold / prefill_ms, 1) if (known and prefill_ms > 0) else 1.0
    return {
        "prefix_id": pid,
        "prefix_warm": known,            # was this prefix seen before?
        "prompt_tokens": prompt_tokens,  # Ollama still reports the full prompt length
        "prefill_ms": prefill_ms,        # but the TIME drops when the prefix is reused
        "prefill_speedup_vs_cold": speedup,
        "total_seconds": round(total, 3),
        "response": data["message"]["content"],
    }


@app.get("/prefix/stats")
def prefix_stats():
    with _lock:
        return _prefixes
