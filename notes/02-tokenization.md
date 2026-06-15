# 02 — Tokenization

> Tokens are the atomic unit of inference. You bill in them, you batch in them, you cache them, and your latency is measured per-token. Get this right first.

## What is a token?

A **token** is a chunk of text — usually a *sub-word*, not a whole word and not a single character. The model never sees text. It sees **integers** (token IDs), each mapping to a learned vector.

```
Text:   "Inference engineering is fun"
Tokens: ["In", "ference", " engineering", " is", " fun"]
IDs:    [  686,   2251,        14667,        374,  2464  ]
        (every token, including the leading space, is its own ID)
```

Rough rule of thumb for English: **1 token ≈ 4 characters ≈ 0.75 words.** 100 tokens ≈ 75 words ≈ a short paragraph.

> Backend analogy: tokenization is your **serialization format**. Just like you encode a struct into bytes before it goes over the wire, the tokenizer encodes text into IDs before it enters the model. And just like with serialization, the *scheme* matters: a bad tokenizer wastes "bytes" (tokens) and costs you money and context window.

---

## Why sub-words? (BPE in 60 seconds)

Two naive options both fail:

- **Word-level**: vocabulary explodes (millions of words, typos, names) and can't handle unseen words.
- **Character-level**: sequences become enormous (every char is a token) → way more forward passes → slow + short effective context.

**Byte-Pair Encoding (BPE)** is the compromise that won. Start from individual bytes, then repeatedly merge the most frequent adjacent pair into a new token. Common words become single tokens; rare words split into pieces; *anything* can be represented (no "unknown word").

```
Start:   "l" "o" "w" "e" "r"
Merge most frequent pairs over a big corpus...
Result:  "low" + "er"   →   "lower" tokenizes as ["low", "er"]
```

This is why `" the"` is one token but a rare name like `"Radadiya"` might be 3–4 tokens. Modern tokenizers (GPT's `tiktoken`, Llama/Qwen's SentencePiece/BPE) all use variants of this.

---

## Why this matters for *inference* (not just NLP trivia)

Everything downstream is counted in tokens, so the tokenizer directly controls cost and speed:

1. **Cost** — APIs bill per token. A verbose tokenizer = more tokens = more $ for identical text. Non-English text often tokenizes 2–3× worse, so the same sentence in Hindi or code can cost multiples of the English version.

2. **Context window** — a "32k context" model means 32k *tokens*, prompt + output combined. Your tokenizer decides how much real text fits.

3. **Latency** — recall from [[01-what-is-inference]]: output tokens = forward passes. `eval_count` (output tokens) is the single biggest driver of end-to-end latency. In POC1, the 33-token answer took 1.27s while the 80-token answers took ~2.7s. **Same model, same machine — latency tracked token count almost linearly.**

4. **Batching & KV cache** — every token gets a KV-cache entry (see [[04-kv-cache]]). Tokens are literally the unit of GPU memory you allocate per request.

---

## See it yourself (Ollama exposes the counts)

Ollama's API returns token counts directly — that's what POC1 reads:

```
prompt_eval_count   →  # of input (prompt) tokens   ← prefill work
eval_count          →  # of output (generated) tokens ← decode work
prompt_eval_duration→  time spent on prefill (ns)
eval_duration       →  time spent on decode (ns)
```

`tokens_per_second = eval_count / (eval_duration / 1e9)` — pure decode speed, independent of prompt length. That's why it stays ~30–34 tok/s across all POC1 prompts even as latency varies: **tokens/sec measures the engine; latency measures the request.**

---

## The gotchas that bite production systems

- **Token ≠ word ≠ character.** Never size buffers, truncation, or cost estimates in characters. Count tokens with the *model's own* tokenizer.
- **Whitespace and casing are tokens too.** `"Hello"`, `" Hello"`, and `"hello"` can be different IDs. This is why prompt formatting subtly changes results.
- **Different models, different tokenizers.** You cannot compare "tokens" across model families 1:1. A Llama token ≠ a GPT token.
- **The chat template is tokens too.** `<|im_start|>user ... <|im_end|>` wrapper tokens are real prompt tokens you pay for on every turn. Qwen, Llama, etc. each have their own special tokens.
- **Output token caps control cost & latency.** POC1 sets `num_predict: 80` — a hard cap on generated tokens. This is your latency seatbelt (`max_tokens` in the OpenAI API).

---

## Connection to real systems

- **vLLM / TGI / TensorRT-LLM** all run the tokenizer on the server before prefill, and de-tokenize incrementally while streaming.
- **Prefix caching** (POC6) works at the *token* level: identical leading token sequences share cached computation.
- **Speculative decoding** (advanced) predicts *multiple tokens* per step to beat the one-token-per-pass limit.

---

### Try this
1. Send a 2-word prompt and a 200-word prompt to POC1. Watch `prompt_eval_count` and TTFT scale with prompt length.
2. Ask for "one word" vs "a long essay" and watch `eval_count` drive total latency while `tokens_per_second` stays flat.

Next: [[03-attention]] — how the model relates these tokens to each other (and why it's expensive).
