"""Classical probability models.  All share the predict()/update() interface."""
from __future__ import annotations

import numpy as np


class Order0:
    """Adaptive frequency counts, no context.  Roughly an adaptive Huffman."""

    name = "order0"

    def __init__(self):
        self.counts = np.ones(256, dtype=np.float64)
        self.total = 256.0

    def predict(self):
        return self.counts / self.total

    def update(self, b):
        self.counts[b] += 1.0
        self.total += 1.0


class ContextMix:
    """Order-k context model with recursive PPM-C style blending.

    p_k(x | ctx_k) = (n_k(x) + e_k * p_{k-1}(x)) / (N_k + e_k),  e_k = alpha * d_k

    where d_k is the number of distinct symbols seen in this context (the
    PPM-C escape estimate).  A context that has produced many different
    symbols is unpredictable and leans on the lower order; one that keeps
    producing the same symbol is trusted.  Contexts are the last k bytes.

    Low orders keep dense 256-entry count arrays.  Higher orders are sparse:
    a context seen with a single successor so far is packed into one int
    (symbol | count << 8), and only contexts with two or more distinct
    successors get a {symbol: count} dict.  Long contexts are overwhelmingly
    deterministic (the PPM* observation), so this keeps memory small.
    """

    name = "ctx"
    DENSE_ORDERS = 2

    def __init__(self, max_order: int = 3, alpha: float = 2.0, discount: float = 0.0, orders=None):
        self.max_order = max_order
        # which context lengths to model; blending walks them in increasing order
        self.orders = list(orders) if orders is not None else list(range(max_order + 1))
        assert self.orders[0] == 0 and self.orders == sorted(self.orders) and self.orders[-1] <= max_order
        self.alpha = alpha
        # discount > 0 switches to interpolated absolute discounting (Kneser-Ney style
        # without continuation counts): p_k = (max(n_k(x)-D, 0) + D*d_k*p_{k-1}(x)) / N_k
        self.discount = discount
        self.dense = [dict() for _ in range(min(max_order, self.DENSE_ORDERS) + 1)]
        self.dense[0][b""] = np.ones(256, dtype=np.float64)
        self.sparse = [dict() for _ in range(max_order + 1)]  # index k >= DENSE_ORDERS+1
        self.hist = b""

    def _ctx(self, k):
        return self.hist[len(self.hist) - k:] if k else b""

    def predict(self):
        p = None
        for k in self.orders:
            if k > len(self.hist):
                break
            ctx = self._ctx(k)
            if k <= self.DENSE_ORDERS:
                counts = self.dense[k].get(ctx)
                if counts is None:
                    break
                n = counts.sum()
                if p is None:
                    p = counts / n
                elif self.discount:
                    d = np.count_nonzero(counts)
                    p = (np.maximum(counts - self.discount, 0.0) + self.discount * d * p) / n
                else:
                    esc = self.alpha * np.count_nonzero(counts)
                    p = (counts + esc * p) / (n + esc)
            else:
                entry = self.sparse[k].get(ctx)
                if entry is None:
                    break
                if isinstance(entry, int):
                    # packed single-successor context: symbol | count << 8
                    sym, c = entry & 0xFF, entry >> 8
                    n, d = c, 1
                    if self.discount:
                        q = p * (self.discount / n)
                        q[sym] += max(c - self.discount, 0.0) / n
                    else:
                        esc = self.alpha
                        q = p * (esc / (n + esc))
                        q[sym] += c / (n + esc)
                    p = q
                    continue
                n, d = entry[-1], len(entry) - 1
                if self.discount:
                    D = self.discount
                    q = p * (D * d / n)
                    for s, c in entry.items():
                        if s != -1:
                            q[s] += max(c - D, 0.0) / n
                else:
                    esc = self.alpha * d
                    q = p * (esc / (n + esc))
                    for s, c in entry.items():
                        if s != -1:
                            q[s] += c / (n + esc)
                p = q
        return p

    def update(self, b):
        for k in self.orders:
            if k > len(self.hist):
                break
            ctx = self._ctx(k)
            if k <= self.DENSE_ORDERS:
                counts = self.dense[k].get(ctx)
                if counts is None:
                    counts = np.zeros(256, dtype=np.float64)
                    self.dense[k][ctx] = counts
                counts[b] += 1.0
            else:
                table = self.sparse[k]
                entry = table.get(ctx)
                if entry is None:
                    table[ctx] = b | (1 << 8)            # packed: first successor, count 1
                elif isinstance(entry, int):
                    if entry & 0xFF == b:
                        table[ctx] = entry + (1 << 8)   # same successor again
                    else:                                # second distinct symbol: expand to a dict
                        c = entry >> 8
                        table[ctx] = {-1: c + 1, entry & 0xFF: c, b: 1}
                else:
                    entry[-1] += 1
                    entry[b] = entry.get(b, 0) + 1
        self.hist = (self.hist + bytes([b]))[-self.max_order:]


MODELS = {
    "order0": lambda: Order0(),
    "ctx2": lambda: ContextMix(max_order=2),
    "ctx3": lambda: ContextMix(max_order=3),
    "ctx4": lambda: ContextMix(max_order=4),
}
