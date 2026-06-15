from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok", "service": "mini-inference-server"}

@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": req.message}
        ],
        "stream": False,
        "options": {
            "num_predict": 80
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    return {
        "model": MODEL,
        "response": data["message"]["content"],
        "latency_seconds": round(time.time() - start, 3),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "tokens_per_second": round(
            data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1_000_000_000),
            2
        ) if data.get("eval_duration") else None
    }