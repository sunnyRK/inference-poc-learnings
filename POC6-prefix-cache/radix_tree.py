"""
POC6 — Prefix Tree (the data structure behind prefix caching)

A minimal trie over "tokens" that models how engines like SGLang
(RadixAttention) and vLLM (automatic prefix caching) detect a SHARED PROMPT
PREFIX across requests, so the model can reuse that prefix's KV cache instead
of recomputing it.

Real engines key on actual model tokens; here we use whitespace-split words as
stand-in "tokens" so the idea is easy to see. The logic is the same:
walk the tree, count how many leading tokens are already stored = the part
whose KV cache can be reused.
"""


class PrefixTree:
    def __init__(self):
        self.root: dict = {}
        self._END = "__end__"

    def insert(self, tokens: list[str]) -> None:
        """Remember a token sequence (e.g. a prompt we've already processed)."""
        node = self.root
        for tok in tokens:
            node = node.setdefault(tok, {})
        node[self._END] = True

    def longest_prefix_match(self, tokens: list[str]) -> int:
        """How many LEADING tokens of `tokens` are already stored? (the reusable part)"""
        node = self.root
        matched = 0
        for tok in tokens:
            if tok in node:
                node = node[tok]
                matched += 1
            else:
                break
        return matched


if __name__ == "__main__":
    tree = PrefixTree()

    # A shared system prompt (the reusable prefix) + two different questions.
    system = "You are a helpful expert programming assistant answer concisely".split()
    request_a = system + "What is Python ?".split()
    request_b = system + "What is Rust ?".split()

    # Request A arrives first — we process and remember it.
    tree.insert(request_a)
    print("Inserted Request A:", " ".join(request_a))

    # Request B arrives — how much can we reuse?
    reused = tree.longest_prefix_match(request_b)
    total = len(request_b)
    print("Request B:         ", " ".join(request_b))
    print()
    print(f"Shared leading tokens : {reused} / {total}")
    print(f"Reusable (cached)     : {' '.join(request_b[:reused])}")
    print(f"Must compute fresh    : {' '.join(request_b[reused:])}")
    print()
    print(f"=> The model can REUSE the KV cache for {reused} tokens and only")
    print(f"   run prefill on the {total - reused} new tokens. That's prefix caching.")
