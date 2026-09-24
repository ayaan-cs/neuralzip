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


class MatchModel:
    """Predict the byte that followed the last occurrence of the current context.

    A hash of the last `min_len` bytes points at the position just after the
    most recent occurrence of that string.  While the prediction keeps coming
    true the match is extended, so a long repeated passage is followed byte by
    byte at almost no cost.  This is the classical "match model" of the PAQ
    family, and the mechanism PPM* relies on: unbounded contexts are usually
    deterministic (Cleary & Teahan 1997).

    The confidence is not assumed, it is learned: hits and misses are counted
    per match length, so the model discovers for itself how much a 6-byte
    match is worth against a 30-byte one.  With no match it returns the
    uniform distribution, which is the honest "no opinion" — and in a
    geometric mixture a constant vector drops out of the softmax entirely,
    so the model costs the mixture nothing when it has nothing to say.
    """

    name = "match"
    MAXLEN = 32  # match lengths are bucketed up to here for the confidence table

    def __init__(self, min_len: int = 6, table_bits: int = 22):
        self.min_len = min_len
        self.mask = (1 << table_bits) - 1
        self.table = np.full(1 << table_bits, -1, dtype=np.int64)
        self.hist = bytearray()
        self.ptr = -1        # index in hist of the byte we are predicting
        self.mlen = 0        # verified length of the current match
        self.hits = np.full(self.MAXLEN + 1, 1.0)
        self.miss = np.full(self.MAXLEN + 1, 1.0)
        self._pred = -1      # byte predicted at the last predict(), -1 if none
        self._bucket = -1
        self._uniform = np.full(256, 1.0 / 256)

    def _hash(self, buf) -> int:
        h = 0
        for c in buf:
            h = (h * 0x2F0FD693 + c + 1) & 0xFFFFFFFF
        return (h ^ (h >> 15)) & self.mask

    def predict(self):
        if self.mlen and 0 <= self.ptr < len(self.hist):
            self._pred = self.hist[self.ptr]
            self._bucket = min(self.mlen, self.MAXLEN)
            h, m = self.hits[self._bucket], self.miss[self._bucket]
            p_hit = min(h / (h + m), 0.9995)
            p = np.full(256, (1.0 - p_hit) / 255.0)
            p[self._pred] = p_hit
            return p
        self._pred, self._bucket = -1, -1
        return self._uniform

    def update(self, b):
        if self._bucket >= 0:
            if self._pred == b:
                self.hits[self._bucket] += 1.0
            else:
                self.miss[self._bucket] += 1.0
        if self.mlen:
            if self.hist[self.ptr] == b:
                self.ptr += 1
                self.mlen = min(self.mlen + 1, self.MAXLEN)
            else:
                self.mlen, self.ptr = 0, -1

        self.hist.append(b)
        n = len(self.hist)
        if n >= self.min_len:
            k = self.min_len
            ctx = self.hist[n - k:n]
            h = self._hash(ctx)
            if not self.mlen:
                cand = int(self.table[h])
                # verify the bytes: a hash collision would predict nonsense
                if cand >= k and self.hist[cand - k:cand] == ctx:
                    self.ptr, self.mlen = cand, k
            self.table[h] = n


MODELS = {
    "order0": lambda: Order0(),
    "ctx2": lambda: ContextMix(max_order=2),
    "ctx3": lambda: ContextMix(max_order=3),
    "ctx4": lambda: ContextMix(max_order=4),
    "match": lambda: MatchModel(),
}
