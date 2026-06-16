"""
POC3 — Streaming Inference Server

Same model as POC1/POC2, but now we can send the answer back token-by-token
*as it is being written*, instead of waiting for the whole thing.

Two endpoints so we can compare them fairly:
  POST /chat          -> NON-streaming: wait for all tokens, then reply once (like POC1)
  POST /chat/stream   -> STREAMING:     send each token the moment the model writes it

Streaming uses Server-Sent Events (SSE) — the same simple "data: ...\\n\\n"
format that OpenAI's streaming API uses. Each event carries one token as JSON.
"""

import json
import time

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
NUM_PREDICT = 80  # hard cap on output tokens (same as POC1, for a fair comparison)


class ChatRequest(BaseModel):
    message: str


def _ollama_payload(message: str, stream: bool) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": message}],
        "stream": stream,
        "options": {"num_predict": NUM_PREDICT},
    }


@app.get("/")
def health():
    return {"status": "ok", "service": "streaming-inference-server"}


@app.post("/chat")
def chat(req: ChatRequest):
    """NON-streaming: the client gets NOTHING until the whole answer is ready."""
    start = time.time()
    response = requests.post(OLLAMA_URL, json=_ollama_payload(req.message, stream=False))
    data = response.json()
    return {
        "mode": "non-streaming",
        "response": data["message"]["content"],
        "latency_seconds": round(time.time() - start, 3),
        "eval_count": data.get("eval_count"),
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """STREAMING: forward each token to the client the instant Ollama produces it."""

    def event_generator():
        payload = _ollama_payload(req.message, stream=True)
        # stream=True on requests keeps the connection open and reads chunks as they arrive
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    # one SSE event per token, token wrapped in JSON so spaces/newlines are safe
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if data.get("done"):
                    # final event carries the token count, then a DONE marker (OpenAI-style)
                    yield f"data: {json.dumps({'done': True, 'eval_count': data.get('eval_count')})}\n\n"
                    yield "data: [DONE]\n\n"
                    break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
