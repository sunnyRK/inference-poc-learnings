"""
POC8 (part 1) — Own the KV cache.

Until now, Ollama created and managed the KV cache for us. Here we run a real
model (distilgpt2, 82M) with HuggingFace transformers and manage the KV cache
OURSELVES: we hold a DynamicCache object, pass it into every forward pass, and
watch it grow by one token each decode step.

We compare two ways to generate the SAME text:
  WITH cache    : feed only the new token each step, reuse stored K/V   -> O(n)
  WITHOUT cache : re-feed the whole sequence each step, recompute all   -> O(n^2)

This is the exact thing the KV cache buys you (POC7 proved it from the outside;
here we prove it from the inside, in our own loop).
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "distilgpt2"
print("loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


@torch.no_grad()
def generate_with_cache(prompt: str, n: int = 60):
    ids = tok(prompt, return_tensors="pt").input_ids
    cache = DynamicCache()  # <-- OUR KV cache, starts empty

    # PREFILL: process the whole prompt once; cache now holds the prompt's K/V.
    out = model(ids, past_key_values=cache, use_cache=True)
    print(f"  [with cache] after prefill, our cache holds {cache.get_seq_length()} tokens")
    next_id = out.logits[:, -1:, :].argmax(-1)
    generated = [next_id.item()]

    t = time.time()
    for _ in range(n - 1):
        # DECODE: feed ONLY the new token; the cache supplies all the history.
        out = model(next_id, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1:, :].argmax(-1)
        generated.append(next_id.item())
    dt = time.time() - t

    print(f"  [with cache] cache grew to {cache.get_seq_length()} tokens "
          f"(prompt + {n} generated)")
    return tok.decode(generated), dt


@torch.no_grad()
def generate_without_cache(prompt: str, n: int = 60):
    ids = tok(prompt, return_tensors="pt").input_ids
    generated = []
    t = time.time()
    for _ in range(n):
        # No cache: recompute attention over the ENTIRE sequence every step.
        out = model(ids, use_cache=False)
        next_id = out.logits[:, -1:, :].argmax(-1)
        generated.append(next_id.item())
        ids = torch.cat([ids, next_id], dim=1)  # sequence keeps growing
    dt = time.time() - t
    return tok.decode(generated), dt


def main():
    prompt = "In the field of machine learning, inference is the process of"
    n = 60

    print(f"\nGenerating {n} tokens two ways (same greedy output expected)\n")

    text_c, t_cache = generate_with_cache(prompt, n)
    text_n, t_nocache = generate_without_cache(prompt, n)

    print("\n" + "=" * 68)
    print("RESULT")
    print("=" * 68)
    print(f"  outputs identical?   {text_c == text_n}")
    print(f"  WITH cache    : {t_cache:.3f} s   ({n / t_cache:.1f} tok/s)")
    print(f"  WITHOUT cache : {t_nocache:.3f} s   ({n / t_nocache:.1f} tok/s)")
    print(f"  => the KV cache made our own loop ~{t_nocache / t_cache:.1f}x faster")
    print(f"\n  sample output: {text_c[:120]!r}")
    print("\n  We held the DynamicCache, passed it each step, and watched it grow.")
    print("  THIS is 'managing the KV cache' — now it's in our code, not Ollama's.")


if __name__ == "__main__":
    main()
