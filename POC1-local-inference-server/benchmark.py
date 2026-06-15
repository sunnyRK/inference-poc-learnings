import requests
import time

URL = "http://localhost:8000/chat"

prompts = [
    "Explain KV cache in simple way",
    "What is batching in LLM inference?",
    "Explain tokenization in 2 lines",
    "What is time to first token?",
    "Why GPU memory matters in inference?"
]

for prompt in prompts:
    start = time.time()

    res = requests.post(URL, json={"message": prompt})
    data = res.json()

    print("\nPrompt:", prompt)
    print("Latency:", data["latency_seconds"], "sec")
    print("Output tokens:", data["eval_count"])
    print("Tokens/sec:", data["tokens_per_second"])