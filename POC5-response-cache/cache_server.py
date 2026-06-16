"""
POC5 — Response Cache

A read-through cache in front of the model. We hash the request into a key;
if we've answered that exact request before, we return the stored answer
instantly (a dict lookup) instead of running the GPU again (~3s).

Same idea as caching an expensive DB query by a hash of the query string.

IMPORTANT — when is caching an LLM safe?
  Caching means "same input -> same output, every time." That's only correct
  if the model is DETERMINISTIC for that input. So we call the model with
  temperature=0 (greedy decoding), which makes the output stable and the
  cache semantically valid. With temperature>0 the model is random on purpose,
  and a cache would wrongly freeze one random answer forever.
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
NUM_PREDICT = 80
CACHE_TTL_SECONDS = None  # None = entries never expire; set e.g. 300 for a 5-min TTL


class ChatRequest(BaseModel):
    message: str


_lock = threading.Lock()
_cache: dict[str, dict] = {}  # key -> {"response", "eval_count", "stored_at"}
_stats = {"hits": 0, "misses": 0}


def cache_key(message: str) -> str:
    """Hash everything that affects the output, so different params never collide."""
    raw = f"{MODEL}|{NUM_PREDICT}|temp=0|{message.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_fresh(entry: dict) -> bool:
    if CACHE_TTL_SECONDS is None:
        return True
    return (time.time() - entry["stored_at"]) < CACHE_TTL_SECONDS


@app.get("/")
def health():
    return {"status": "ok", "service": "response-cache-server"}


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    key = cache_key(req.message)

    # 1. Try the cache first (the fast path).
    with _lock:
        entry = _cache.get(key)
        if entry and _is_fresh(entry):
            _stats["hits"] += 1
            return {
                "response": entry["response"],
                "eval_count": entry["eval_count"],
                "cached": True,
                "latency_seconds": round(time.time() - start, 5),
            }
        _stats["misses"] += 1

    # 2. Cache miss -> actually run the model (temperature=0 so it's cacheable).
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": req.message}],
        "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0},
    }
    data = requests.post(OLLAMA_URL, json=payload).json()
    answer = data["message"]["content"]
    eval_count = data.get("eval_count")

    # 3. Store the result for next time.
    with _lock:
        _cache[key] = {"response": answer, "eval_count": eval_count, "stored_at": time.time()}

    return {
        "response": answer,
        "eval_count": eval_count,
        "cached": False,
        "latency_seconds": round(time.time() - start, 5),
    }


@app.get("/cache/stats")
def cache_stats():
    with _lock:
        total = _stats["hits"] + _stats["misses"]
        hit_rate = (_stats["hits"] / total * 100) if total else 0
        return {
            "hits": _stats["hits"],
            "misses": _stats["misses"],
            "entries": len(_cache),
            "hit_rate_percent": round(hit_rate, 1),
        }


@app.post("/cache/clear")
def cache_clear():
    with _lock:
        _cache.clear()
        _stats["hits"] = 0
        _stats["misses"] = 0
    return {"cleared": True}
