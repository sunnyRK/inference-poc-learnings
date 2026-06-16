"""
POC7 — KV Cache Observer & Verifier

You cannot read Ollama's raw KV-cache blocks (only vLLM/TensorRT-LLM expose
that). But Ollama returns timing metrics we can use to PROVE the KV cache is
working, by experiment.

The fingerprint of a working KV cache:
  - PREFILL time grows a lot as the prompt gets longer (process N tokens once).
  - DECODE speed (tok/s) stays ~FLAT regardless of prompt length, BECAUSE each
    new token reuses the stored K/V instead of re-reading the whole prompt.

If there were NO KV cache, decode would re-process the full context for every
token, so decode tok/s would collapse on long prompts. So:
      flat decode tok/s  +  growing prefill  =  KV cache confirmed.

Metrics used (Ollama returns durations in NANOSECONDS):
  prompt_eval_count    = input  tokens          (prefill)
  prompt_eval_duration = time to process prompt (prefill time)
  eval_count           = output tokens          (decode)
  eval_duration        = time to generate them  (decode time)
"""

import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"

# ~filler we repeat to grow the prompt length on demand.
FILLER = ("KV cache is created inside the attention layers. During prefill the model "
          "processes all prompt tokens and stores key and value tensors for each layer. ")

# A fixed generation task so every run decodes a similar number of output tokens.
TASK = "\n\nIgnore the text above. Write a long detailed essay about computers."


def ns_to_s(ns) -> float:
    return (ns or 0) / 1_000_000_000


def call(prompt: str, num_predict: int = 80, num_ctx: int = 8192) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",  # keep model resident so we don't pay reload cost mid-experiment
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    return requests.post(OLLAMA_URL, json=payload, timeout=300).json()


def measure(filler_repeats: int) -> dict:
    prompt = (FILLER * filler_repeats) + TASK
    d = call(prompt)
    in_tok = d.get("prompt_eval_count", 0)
    out_tok = d.get("eval_count", 0)
    prefill = ns_to_s(d.get("prompt_eval_duration"))
    decode = ns_to_s(d.get("eval_duration"))
    return {
        "in_tok": in_tok,
        "prefill_s": prefill,
        "prefill_tps": in_tok / prefill if prefill else 0,
        "out_tok": out_tok,
        "decode_s": decode,
        "decode_tps": out_tok / decode if decode else 0,
    }


def main():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Ollama not reachable on :11434 — is it running? ({e})")

    print("Warming up (load model)...")
    call("Say ready.", num_predict=3)

    # Sweep prompt length from tiny to large; output length stays fixed at 80 tokens.
    repeats = [0, 6, 15, 30, 60]
    print("\nSweeping prompt length (output fixed at 80 tokens each)...\n")
    print(f"{'in_tok':>7} | {'prefill_s':>9} | {'prefill_tps':>11} || "
          f"{'out_tok':>7} | {'decode_s':>8} | {'decode_tps':>10}")
    print("-" * 78)
    rows = []
    for r in repeats:
        m = measure(r)
        rows.append(m)
        print(f"{m['in_tok']:>7} | {m['prefill_s']:>9.3f} | {m['prefill_tps']:>11.1f} || "
              f"{m['out_tok']:>7} | {m['decode_s']:>8.3f} | {m['decode_tps']:>10.2f}")

    # ---- the verdict ----
    small, large = rows[0], rows[-1]
    prefill_growth = (large["prefill_s"] / small["prefill_s"]) if small["prefill_s"] else 0
    tok_growth = (large["in_tok"] / small["in_tok"]) if small["in_tok"] else 0
    decode_ratio = (large["decode_tps"] / small["decode_tps"]) if small["decode_tps"] else 0

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  prompt grew         : {small['in_tok']} -> {large['in_tok']} tokens (~{tok_growth:.0f}x)")
    print(f"  prefill time grew   : {small['prefill_s']:.3f}s -> {large['prefill_s']:.3f}s (~{prefill_growth:.0f}x)")
    print(f"  decode speed change : {small['decode_tps']:.1f} -> {large['decode_tps']:.1f} tok/s "
          f"({decode_ratio*100:.0f}% of the small-prompt speed)")
    print()
    if decode_ratio > 0.6:
        print("  => Decode speed stayed ~FLAT while prefill grew a lot.")
        print("     The model did NOT re-process the long prompt for each output token.")
        print("     ==> KV CACHE CONFIRMED. Prefill builds it once; decode reuses it.")
    else:
        print("  => Decode speed dropped sharply with prompt length — would suggest NO KV reuse.")


if __name__ == "__main__":
    main()
