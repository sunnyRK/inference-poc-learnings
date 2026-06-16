# 09 — Streaming & TTFT (simple guide)

> POC3 in plain words. Same easy style as [[08-understanding-the-numbers]].

---

## The shop story

In POC1, the cook makes your **whole plate**, then brings it. You sit at an empty table for ~3 seconds, then *bang* — everything arrives at once.

**Streaming** is the cook bringing you **each bite the moment it's ready**. First bite in about half a second, then the next, then the next, until the plate is done.

- The **total cooking time is the same** either way.
- But streaming **feels** much faster, because you're not staring at an empty table.

That's the whole idea. ChatGPT does this — it types words out one by one instead of freezing and then dumping a big block of text.

---

## The one new word: TTFT

**TTFT = Time To First Token = "how long until I see the first word."**

(A **token** = a small piece of a word. The model writes one small piece at a time.)

```
NON-STREAMING:   ask ──[ blank screen for ~3 sec ]── whole answer pops up
                 first word seen at: ~3 sec   ← you wait the whole thing

STREAMING:       ask ─[~0.3 sec]─ word word word word word ...
                 first word seen at: ~0.3 sec ← almost instant
```

Lower TTFT = the answer *starts* sooner = it feels faster.

---

## What we measured (real numbers, Apple M4)

```
mode          first word (TTFT)   whole answer (total)
─────────────────────────────────────────────────────
non-streaming      2.94 sec            ~2.9 sec
streaming          0.31 sec            ~3.1 sec
```

Read it like this:

1. **First word:** 2.94 sec vs **0.31 sec**. With streaming the first word shows up about **9 times sooner**. 🎉
2. **Whole answer:** about **3 seconds either way**. Streaming did **NOT** make the model faster — the full reply still takes the same time.
3. So streaming only changes **when you start seeing words**, not how long the whole thing takes. And "when you start seeing words" is exactly what makes it *feel* fast.

---

## Why this is a big deal

- A 3-second blank screen feels broken. Words appearing in 0.3 sec feels alive. **Same model, totally different feeling.**
- It's almost **free**. In [[07-poc2-learnings]] we saw we can't easily make the GPU faster on this laptop. Streaming makes the experience better *without* needing more speed or more hardware.
- **Every real chat app streams** — OpenAI, Anthropic, Gemini. We even used the same "data: ..." message format that OpenAI uses.

---

## How it works (very short)

1. We ask Ollama to stream (`"stream": true`). Ollama sends back **one small message per token** instead of one big message at the end.
2. Our server passes each token straight to the user the instant it arrives.
3. The user's screen fills up word by word.

Think of it as the cook shouting "bite ready!" after each bite, instead of staying silent until the whole plate is done.

---

## The 2 words to remember

- **Streaming** = send each word as it's made (don't wait for the whole answer).
- **TTFT** = time until the first word shows up. Streaming makes it tiny.

Related: [[08-understanding-the-numbers]] (latency, p50, etc.) · [[01-what-is-inference]] (prefill vs decode — prefill is what TTFT mostly waits on) · [[05-batching]] (real systems stream *and* batch together).
