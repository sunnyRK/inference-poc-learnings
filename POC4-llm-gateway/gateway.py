"""
POC4 — LLM Gateway

A reverse-proxy / API-gateway that sits IN FRONT of the model backend (Ollama)
and enforces the production concerns the model server shouldn't handle itself:

  1. AUTH         - require a valid API key (Bearer token)        -> 401 if missing/invalid
  2. ROUTING      - map a model alias ("fast"/"smart") to a real backend + config
  3. AUTHORIZATION- check the key is allowed to use that model    -> 403 if not
  4. RATE LIMIT   - cap requests per key per minute               -> 429 if exceeded
  5. METRICS      - count requests/tokens per key, measure gateway overhead

Same idea as Kong / Envoy / an API gateway in front of microservices,
but the upstream service is an LLM.
"""

import threading
import time
from collections import defaultdict, deque

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- config: API keys. In production this lives in a DB / secrets store. ---
API_KEYS = {
    "sk-free-001": {"name": "free-tier", "rate_limit_per_min": 5, "allowed_models": ["fast"]},
    "sk-pro-002": {"name": "pro-tier", "rate_limit_per_min": 60, "allowed_models": ["fast", "smart"]},
}

# --- config: model registry. Alias -> real backend + per-model defaults. ---
# We only have qwen2.5:3b locally, so both aliases point to it with different
# token caps. In production "smart" would be a bigger model on a different GPU pool.
MODELS = {
    "fast": {"ollama_model": "qwen2.5:3b", "num_predict": 80},
    "smart": {"ollama_model": "qwen2.5:3b", "num_predict": 200},
}

# --- in-memory state (a real gateway would use Redis so it works across replicas) ---
_lock = threading.Lock()  # endpoints run in a threadpool, so guard shared dicts
_req_times = defaultdict(deque)  # key -> timestamps of recent requests (sliding window)
_stats = defaultdict(lambda: {"ok": 0, "rate_limited": 0, "tokens": 0})


class ChatRequest(BaseModel):
    message: str
    model: str = "fast"


def authenticate(authorization: str | None) -> tuple[str, dict]:
    """Pull the Bearer token out of the Authorization header and validate it."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing 'Authorization: Bearer <key>' header")
    key = authorization.split(" ", 1)[1].strip()
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key, API_KEYS[key]


def check_rate_limit(key: str, limit: int) -> None:
    """Sliding-window rate limit: count this key's requests in the last 60 seconds."""
    now = time.time()
    with _lock:
        times = _req_times[key]
        while times and now - times[0] > 60:  # evict timestamps older than the window
            times.popleft()
        if len(times) >= limit:
            _stats[key]["rate_limited"] += 1
            retry_in = round(60 - (now - times[0]), 1)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit of {limit}/min exceeded. Retry in ~{retry_in}s",
            )
        times.append(now)  # record this request


@app.get("/")
def health():
    return {"status": "ok", "service": "llm-gateway"}


@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys())}


@app.get("/stats")
def stats():
    """Per-key usage counters (named by tier for readability)."""
    return {API_KEYS[k]["name"]: v for k, v in _stats.items()}


@app.post("/v1/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    gateway_start = time.time()

    # 1. AUTH
    key, key_cfg = authenticate(authorization)

    # 2. ROUTING + 3. AUTHORIZATION
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")
    if req.model not in key_cfg["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail=f"Key '{key_cfg['name']}' is not allowed to use model '{req.model}'",
        )
    model_cfg = MODELS[req.model]

    # 4. RATE LIMIT
    check_rate_limit(key, key_cfg["rate_limit_per_min"])

    # 5. FORWARD to the backend, timing the upstream call separately
    payload = {
        "model": model_cfg["ollama_model"],
        "messages": [{"role": "user", "content": req.message}],
        "stream": False,
        "options": {"num_predict": model_cfg["num_predict"]},
    }
    backend_start = time.time()
    r = requests.post(OLLAMA_URL, json=payload)
    backend_time = time.time() - backend_start
    data = r.json()

    eval_count = data.get("eval_count") or 0
    with _lock:
        _stats[key]["ok"] += 1
        _stats[key]["tokens"] += eval_count

    total = time.time() - gateway_start
    return {
        "model": req.model,
        "backend_model": model_cfg["ollama_model"],
        "key": key_cfg["name"],
        "response": data["message"]["content"],
        "eval_count": eval_count,
        "backend_seconds": round(backend_time, 3),
        # everything the gateway added on top of the raw model call:
        "gateway_overhead_ms": round((total - backend_time) * 1000, 2),
    }
