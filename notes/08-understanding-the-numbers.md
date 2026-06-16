# 08 — Understanding the Numbers (a slow, simple guide)

> A beginner-friendly note. We start with a simple shop story, then learn each tech word and its little bit of math — one small step at a time, with tiny easy numbers. No big math words without a plain meaning next to them.

---

## Part 0 — The shop story (keep this picture in your head)

Your server is a small **food shop with one cook**. The cook is the **GPU** (the chip that does the real work).

- A **customer** = one request (someone asking the model a question).
- The **food** = the answer the model writes.
- The **cook** can really make food for **one person at a time**.

POC2 sends a **crowd of customers at once** (1, then 2, then 4, then 8) and writes down two things: *how long each customer waited*, and *how much food the shop made per second*. That's it. Every number below is just measuring one of those two things.

---

## Part 1 — Latency = "how long one customer waits"

**Latency** is just a fancy word for **waiting time** — how long *one* request takes from start to finish. Smaller is better.

How we measure it: start a stopwatch when we send the question, stop it when the answer comes back. Then **subtract**:

```
latency = end time − start time
```

Tiny example: you send the question at second **2**, the answer arrives at second **5**.

```
5 − 2 = 3 seconds   →  latency = 3 seconds
```

That's all latency is. In the code, `time.time()` is the stopwatch, and `time.time() - start` is this subtraction.

---

## Part 2 — Throughput = "how much the shop makes per second"

**Throughput** = how much work the shop finishes **in one second**. Bigger is better. The little word **"per"** means **"in one"** — so "5 plates per second" = "5 plates in one second."

To get a "per second" number, we **divide** (the ÷ sign means *share equally into groups*):

```
throughput = total work ÷ total seconds
```

Tiny example: the shop made **8 plates** in **4 seconds**. How many in *one* second?

```
8 ÷ 4 = 2   →  2 plates per second
```

In POC2 we use two kinds of throughput:
- **req/s** (requests per second) = how many full answers finished each second.
- **tok/s** (tokens per second) = how many small word-pieces the model wrote each second.

> A **token** is a small piece of a word. The model writes text one small piece at a time (like writing "in-fer-ence" piece by piece). So counting tokens is a fairer way to measure "how much work," because some answers are long and some are short.

Tiny example: the model wrote **240** tokens in **8** seconds → `240 ÷ 8 = 30` → **30 tok/s**.

---

## Part 3 — Average (also called "mean") = "the typical-looking middle, by sharing"

**Average** answers: "if everyone waited the *same* amount, what would it be?" You **add all the waits**, then **divide by how many people**.

```
average = (add all the numbers) ÷ (how many numbers)
```

Tiny example: three people waited **2, 3, 4** seconds.

```
add:    2 + 3 + 4 = 9
count:  3 people
share:  9 ÷ 3 = 3   →  average wait = 3 seconds
```

Easy. But average has a **trap** — see the next part.

---

## Part 4 — Why the average can fool you (important!)

Imagine 5 people waited: **2, 2, 2, 2, 20** seconds. (Four were fast, one was very slow.)

```
add:    2 + 2 + 2 + 2 + 20 = 28
share:  28 ÷ 5 = 5.6   →  average = 5.6 seconds
```

But look closely: **4 out of 5 people only waited 2 seconds!** Only **one** unlucky person waited 20. The average (5.6) makes it look like *everyone* had a slow-ish time — but that's **not true**. One big slow number **pulled the average up** and **hid the truth**.

This is why engineers don't trust the average alone. They use the **median** and **percentiles** instead. 👇

---

## Part 5 — Median = "p50" = "the real middle one"

To find the **median**, you **line up all the numbers from small to big**, then pick the **one in the middle**.

Same 5 people, sorted small → big:

```
2   2   2   2   20
        ↑
     middle one = 2   →  median = 2 seconds
```

So the **median is 2 seconds** — this tells the truth: *the normal customer waited 2 seconds.* The one slow person doesn't ruin this number.

**p50 is just another name for the median.** Read it like this:
- **"p"** = position in the sorted line.
- **"50"** = halfway (50 out of 100).
- So **p50** = "half of the customers waited *less* than this, half waited *more*." = the normal, typical wait.

---

## Part 6 — p95 and p99 = "the slow, unlucky customers"

After lining everyone up small → big:

- **p95** = the wait where **95 out of 100** people waited *less* than this. Only the slowest **5** were worse. This is "a pretty bad day."
- **p99** = **99 out of 100** waited less. Only the **1** slowest person was worse. This is "the worst, unluckiest customer."

Picture the sorted line of waits:

```
fast ──────────────────────────────────► slow
2  2  2  2  3  3  3  4  5  ...  9  18  18
                              ↑        ↑
                             p95      p99   ← the slow tail (unlucky people)
```

**Why do we care about the unlucky 1%?** Because real shops serve *millions* of customers. If 1 out of every 100 has a bad time, that is still a **huge** number of unhappy people. Also, one slow web page often needs many requests — if even one of them is slow, the whole page feels slow. So the slow tail (p95, p99) is what people actually complain about. That's why engineers watch it closely.

> Quick name for it: the slow end of the line is called the **"tail."** When someone says "the tail blew up," they mean the slow-unlucky waits (p95/p99) got much worse.

---

## Part 7 — "x times" (like 1.06x or 10x) = "how many times bigger"

When we compare two numbers, we **divide the new by the old** to see how many *times* bigger it got. The answer is called a **ratio**.

```
how many times bigger = new number ÷ old number
```

- **2x** = "two times" = double.
- **10x** = "ten times" = ten times as much.
- **1.06x** = "almost the same" — only a tiny bit (6%) more.

POC2 example: throughput went from **28.3** (1 user) up to **30.1** (4 users).

```
30.1 ÷ 28.3 = 1.06   →  only 1.06x  →  basically NO improvement
```

So even with 4× more customers, the shop made almost the **same** amount of food. That tiny `1.06x` is the proof that "more customers did **not** help."

---

## Part 8 — Now read the real POC2 table like an engineer

```
people at once │  tok/s   │   p50    │   p95    │   p99
   (load)      │ (speed)  │ (normal) │  (slow)  │ (worst)
───────────────┼──────────┼──────────┼──────────┼─────────
      1        │  28.3    │  2.72 s  │  2.95 s  │  3.00 s
      2        │  29.9    │  4.50 s  │  5.51 s  │  5.63 s
      4        │  30.1    │  8.92 s  │  9.32 s  │  9.40 s
      8        │  28.9    │ 11.14 s  │ 18.18 s  │ 18.40 s
```

Read it in plain words:

1. **Look down the `tok/s` column:** 28 → 30 → 30 → 29. It **barely changes.** The cook's speed never went up. → *More customers did not make the shop faster.*

2. **Look down the `p50` column:** 2.7 → 4.5 → 8.9 → 11.1. The normal wait keeps **doubling** as we double the crowd. → *Each customer just waits longer in line.*

3. **Compare `p50` and `p99` in the last row:** normal customer waited 11s, but the unlucky one waited **18.4s**. → *The slow tail got much worse than the normal wait.*

**The whole story in one sentence:** adding more customers to one cook did **not** make more food per second — it only made people **wait longer**. To actually serve a crowd faster, the cook needs a smarter trick called **batching** (cooking for many people in one go) — that's what later POCs build.

---

### The 4 words to remember
- **Latency** = one person's waiting time (use *subtraction*).
- **Throughput** = work finished per second (use *division*).
- **p50 (median)** = the normal, middle experience (use *sorting, pick the middle*).
- **p95 / p99** = the slow, unlucky people — the "tail."

Related: [[07-poc2-learnings]] (the POC2 results) · [[05-batching]] (the "cook for many at once" trick) · [[01-what-is-inference]] (what a token is).
