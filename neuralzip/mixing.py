"""Combine several predictors into one (better) predictor.

Bayesian mixture with discounting ("fixed share" style):

    p(x)   = sum_i w_i p_i(x)
    L_i   <- gamma * L_i - log p_i(x)          discounted cumulative log-loss
    w_i    = softmax(-beta * L_i), floored at eps/N

With gamma = 1, beta = 1 this is exactly the Bayesian posterior over experts,
and the code length of the mixture is at most log2(N) bits worse than the
best single expert on the whole file.  gamma < 1 forgets the distant past so
the weights can track whichever expert is currently best (e.g. the context
model early on, the neural net once it has learned something).
"""
from __future__ import annotations

import numpy as np


class Mixture:
    name = "mix"

    def __init__(self, experts, beta=1.0, gamma=0.995, floor=1e-3):
        self.experts = list(experts)
        self.beta, self.gamma, self.floor = beta, gamma, floor
        self.L = np.zeros(len(self.experts))
        self._preds = None

    def weights(self):
        w = np.exp(-self.beta * (self.L - self.L.min()))
        w /= w.sum()
        n = len(w)
        w = (1 - self.floor) * w + self.floor / n
        return w

    def predict(self):
        self._preds = [e.predict() for e in self.experts]
        w = self.weights()
        p = np.zeros(256)
        for wi, pi in zip(w, self._preds):
            p += wi * pi
        return p

    def update(self, b):
        for i, pi in enumerate(self._preds):
            self.L[i] = self.gamma * self.L[i] - np.log(max(pi[b], 1e-12))
        for e in self.experts:
            e.update(b)


class GeoMixture:
    """Log-linear ("geometric") mixing with weights learned online.

        p(x) = softmax_x( sum_i w_i * log p_i(x) )

    The weights are updated by gradient descent on the code length -log p(x):

        dL/dw_i = E_p[log p_i] - log p_i(x)

    Unlike a linear mixture, weights are not constrained to sum to one, so
    when independent experts agree the mixture can become *sharper* than
    either of them.  This is the multi-symbol analogue of the logistic
    mixing used by the PAQ family of compressors.

    Optionally the weight vector is selected by a small context (here: the
    previous byte's class), so the mixer can learn e.g. that the neural net
    is more trustworthy after a letter than after punctuation.
    """

    name = "geomix"

    def __init__(self, experts, lr=0.02, init=None, n_ctx=1, ctx_fn=None):
        self.experts = list(experts)
        n = len(self.experts)
        self.lr = lr
        w0 = np.full(n, 1.0 / n) if init is None else np.asarray(init, dtype=np.float64)
        self.W = np.tile(w0, (n_ctx, 1))
        self.ctx_fn = ctx_fn or (lambda prev: 0)
        self.prev = 256
        self._logs = None
        self._p = None
        self._ctx = 0

    def weights(self):
        return self.W[self._ctx]

    def predict(self):
        self._logs = np.stack([np.log(np.maximum(e.predict(), 1e-12)) for e in self.experts])  # (n, 256)
        self._ctx = self.ctx_fn(self.prev)
        s = self.W[self._ctx] @ self._logs
        s -= s.max()
        p = np.exp(s)
        p /= p.sum()
        self._p = p
        return p

    def update(self, b):
        # gradient of -log p(b) wrt each weight
        grad = self._logs @ self._p - self._logs[:, b]
        self.W[self._ctx] -= self.lr * grad
        self.prev = b
        for e in self.experts:
            e.update(b)


def byte_class(prev: int) -> int:
    """Coarse class of the previous byte: 0 letter, 1 digit, 2 space/newline, 3 other."""
    if prev == 256:
        return 3
    c = chr(prev)
    if c.isalpha():
        return 0
    if c.isdigit():
        return 1
    if c in " \n\t\r":
        return 2
    return 3
