"""
POC8 (part 2) — Mini-vLLM: continuous batching with our own scheduler.

We run a real model (distilgpt2) and compare two ways to serve a queue of
requests that have DIFFERENT output lengths:

  SEQUENTIAL          : finish one request fully, then start the next.
  CONTINUOUS BATCHING : keep up to `capacity` requests decoding together; the
                        instant one finishes, evict it and admit a waiting one,
                        so the batch stays full and the GPU stays busy.

This is the core idea of vLLM / TGI (note 05). Our scheduler owns:
  - the KV cache (a shared DynamicCache for the active batch)
  - admission (pull from the queue when a slot frees)
  - eviction (drop a finished request)

Simplification vs real vLLM: when the batch composition changes (a request
finishes or joins), we REBUILD the batched KV cache by re-prefilling the active
sequences. Real engines avoid that re-prefill with PagedAttention (they page the
KV cache instead of recomputing it). We call that out honestly.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "distilgpt2"
print("loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
PAD = tok.pad_token_id


class Request:
    def __init__(self, rid, prompt, max_tokens):
        self.rid = rid
        self.max_tokens = max_tokens
        self.token_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        self.n_generated = 0
        self.done = False

    def append(self, token_id):
        self.token_ids.append(token_id)
        self.n_generated += 1
        if self.n_generated >= self.max_tokens or token_id == tok.eos_token_id:
            self.done = True


# ----------------------------- SEQUENTIAL -----------------------------
@torch.no_grad()
def run_sequential(requests):
    total_tokens = 0
    t0 = time.time()
    for r in requests:
        ids = torch.tensor([r.token_ids])
        cache = DynamicCache()
        out = model(ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:, :].argmax(-1)
        for _ in range(r.max_tokens):
            total_tokens += 1
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:, :].argmax(-1)
    return time.time() - t0, total_tokens


# ------------------------- CONTINUOUS BATCHING ------------------------
@torch.no_grad()
def _prefill_batch(active):
    """Rebuild a shared KV cache by prefilling all active sequences (left-padded)."""
    seqs = [r.token_ids for r in active]
    lmax = max(len(s) for s in seqs)
    a = len(active)
    ids = torch.full((a, lmax), PAD, dtype=torch.long)
    mask = torch.zeros((a, lmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, lmax - len(s):] = torch.tensor(s)
        mask[i, lmax - len(s):] = 1
    pos = mask.cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    cache = DynamicCache()
    out = model(ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1)           # [A]
    return nxt, cache, mask, pos[:, -1]             # next tokens, cache, mask, last positions


@torch.no_grad()
def _decode_step(last_tokens, cache, mask, last_pos):
    """One cheap decode step over the active batch, reusing the shared cache."""
    a = last_tokens.size(0)
    mask = torch.cat([mask, torch.ones((a, 1), dtype=mask.dtype)], dim=1)
    pos = (last_pos + 1).unsqueeze(1)
    out = model(last_tokens.unsqueeze(1), attention_mask=mask,
                position_ids=pos, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1)
    return nxt, cache, mask, last_pos + 1


@torch.no_grad()
def run_continuous_batching(requests, capacity):
    queue = list(requests)
    active = []
    total_tokens = 0
    rebuilds = 0
    cache = mask = last_pos = last_tokens = None
    need_rebuild = True

    t0 = time.time()
    while queue or active:
        # 1. EVICT finished requests
        if any(r.done for r in active):
            active = [r for r in active if not r.done]
            need_rebuild = True
        # 2. ADMIT waiting requests into free slots
        while len(active) < capacity and queue:
            active.append(queue.pop(0))
            need_rebuild = True
        if not active:
            break

        # 3. one forward pass (rebuild on composition change, else cheap decode)
        if need_rebuild:
            last_tokens, cache, mask, last_pos = _prefill_batch(active)
            need_rebuild = False
            rebuilds += 1
        else:
            last_tokens, cache, mask, last_pos = _decode_step(last_tokens, cache, mask, last_pos)

        # 4. record the generated token for each active request
        for i, r in enumerate(active):
            r.append(int(last_tokens[i]))
            total_tokens += 1

    return time.time() - t0, total_tokens, rebuilds


def make_workload():
    # 8 requests with DIFFERENT output lengths -> finish at different times,
    # which is exactly where continuous batching beats waiting.
    prompts = [
        "The history of computing began",
        "Artificial intelligence is",
        "In the future, software will",
        "The most important idea in physics is",
        "Once upon a time there was",
        "To make good coffee you should",
        "The ocean is full of",
        "Learning to code means",
    ]
    lengths = [20, 40, 25, 55, 30, 35, 22, 48]
    return [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, lengths))]


def main():
    seq_time, seq_tokens = run_sequential(make_workload())

    print("\n" + "=" * 70)
    print("MINI-vLLM — throughput vs batch capacity")
    print("=" * 70)
    print("  workload: 8 requests, output lengths 20-55 tokens "
          f"({seq_tokens} tokens total)\n")
    print(f"  {'mode':<26} {'time':>7} {'tok/s':>8} {'vs seq':>8}")
    print("  " + "-" * 52)
    print(f"  {'sequential (capacity=1)':<26} {seq_time:>6.2f}s {seq_tokens / seq_time:>7.1f} "
          f"{'1.0x':>8}")

    for cap in (2, 4, 8):
        cb_time, cb_tokens, rebuilds = run_continuous_batching(make_workload(), capacity=cap)
        print(f"  {'continuous (capacity=' + str(cap) + ')':<26} {cb_time:>6.2f}s "
              f"{cb_tokens / cb_time:>7.1f} {seq_time / cb_time:>7.1f}x")

    print("\n  => higher batch capacity = more requests share each forward pass")
    print("     = higher throughput. Same model, same work; only the scheduler")
    print("     changed. This is the core win of vLLM/TGI continuous batching.")


if __name__ == "__main__":
    main()
