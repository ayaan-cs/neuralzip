"""A byte-level GRU language model that trains itself while it compresses.

Nothing is pre-trained and no weights are stored in the archive.  The model
starts from a fixed random seed, predicts the next byte, is told the answer,
and every `bptt` bytes runs truncated back-propagation through time and an
Adam step.  The decoder performs the exact same sequence of floating point
operations, so it reconstructs the same weights and the same probabilities.

Everything is float32 and single-threaded (see __init__.py) so that the
encoder and decoder are bit-identical.
"""
from __future__ import annotations

import numpy as np

START = 256  # extra token for "before the first byte"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class GRUByteModel:
    name = "gru"

    def __init__(self, hidden=128, embed=64, bptt=16, lr=3e-3, clip=1.0, passes=1, lr_decay=3000,
                 beta1=0.9, beta2=0.999, eps=1e-8, lr_linear_to=None, total_steps=None,
                 retrain_every=0, retrain_window=0, retrain_lr_scale=0.5, seed=0, dtype=np.float32):
        self.H, self.D, self.T, self.lr, self.clip = hidden, embed, bptt, lr, clip
        self.passes = passes  # extra gradient steps on the same window (still causal)
        self.lr_decay = lr_decay  # if > 0: lr_t = lr / sqrt(1 + step / lr_decay)
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        H, D = hidden, embed

        def u(shape, scale):
            return rng.uniform(-scale, scale, shape).astype(dtype)

        # Parameters.  Gates are stacked [z | r | h] along the last axis.
        self.P = {
            "E": u((257, D), 0.5),
            "W": u((D, 3 * H), 1 / np.sqrt(D)),
            "U": u((H, 3 * H), 1 / np.sqrt(H)),
            "b": np.zeros(3 * H, dtype=dtype),
            "Wo": u((H, 256), 1 / np.sqrt(H)),
            "bo": np.zeros(256, dtype=dtype),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.step = 0
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        # optional NNCP-style linear decay from lr to lr_linear_to over total_steps updates
        self.lr_linear_to, self.total_steps = lr_linear_to, total_steps

        self.h = np.zeros(H, dtype=dtype)
        self.prev = START
        self.cache = []  # per-step activations kept for BPTT
        self._step = None
        # NNCP-v2-style periodic retraining on data already seen (Bellard 2021):
        # every `retrain_every` bytes, run one extra training pass over the last
        # `retrain_window` bytes at a reduced learning rate.  Still causal: the
        # decoder has those bytes too.
        self.retrain_every, self.retrain_window, self.retrain_lr_scale = retrain_every, retrain_window, retrain_lr_scale
        self.history = bytearray() if retrain_every else None
        self.seen = 0
        self._retraining = False

    # -- forward ------------------------------------------------------------
    def predict(self):
        P, H = self.P, self.H
        x = P["E"][self.prev]
        a = x @ P["W"] + P["b"]
        hU = self.h @ P["U"]
        z = _sigmoid(a[:H] + hU[:H])
        r = _sigmoid(a[H:2 * H] + hU[H:2 * H])
        rh = r * self.h
        hc = np.tanh(a[2 * H:] + rh @ P["U"][:, 2 * H:])
        h_new = (1.0 - z) * self.h + z * hc
        logits = h_new @ P["Wo"] + P["bo"]
        logits = logits - logits.max()
        e = np.exp(logits)
        p = e / e.sum()
        self._step = (self.prev, self.h, z, r, rh, hc, h_new, p)
        return p.astype(np.float64)

    def update(self, b):
        tok_in, h_prev, z, r, rh, hc, h_new, p = self._step
        self.cache.append((tok_in, h_prev, z, r, rh, hc, h_new, p, b))
        self.h = h_new
        self.prev = b
        if len(self.cache) >= self.T:
            self._adam(self._gradients())
            for _ in range(self.passes - 1):
                self._replay_window()
                self._adam(self._gradients())
            self.cache = []
        if self.history is not None and not self._retraining:
            self.history.append(b)
            self.seen += 1
            if self.seen % self.retrain_every == 0:
                self._retrain()

    def _retrain(self):
        window = bytes(self.history[-self.retrain_window:])
        saved = (self.h, self.prev, self.cache, self.lr)
        self._retraining = True
        self.h = np.zeros(self.H, dtype=self.dtype)
        self.prev = START
        self.cache = []
        self.lr = saved[3] * self.retrain_lr_scale
        for b in window:
            self.predict()
            self.update(b)
        self.h, self.prev, self.cache, self.lr = saved
        self._retraining = False

    def _replay_window(self):
        """Re-run the forward pass over the cached window with the updated weights.

        Starts from the window's original initial state so the extra pass is a
        genuine second gradient step on the same data.  The final hidden state
        is then the one the *updated* model produces, and is carried forward.
        """
        tokens = [c[8] for c in self.cache]
        self.h = self.cache[0][1]
        self.prev = self.cache[0][0]
        self.cache = []
        for b in tokens:
            self.predict()
            tok_in, h_prev, z, r, rh, hc, h_new, p = self._step
            self.cache.append((tok_in, h_prev, z, r, rh, hc, h_new, p, b))
            self.h = h_new
            self.prev = b

    # -- backward -----------------------------------------------------------
    def _gradients(self):
        """Truncated BPTT over the cached window; returns dLoss/dParam (mean NLL)."""
        P, H = self.P, self.H
        G = {k: np.zeros_like(v) for k, v in P.items()}
        Uz, Ur, Uh = P["U"][:, :H], P["U"][:, H:2 * H], P["U"][:, 2 * H:]
        T = len(self.cache)

        # Output layer is batched across the window.
        Hs = np.stack([c[6] for c in self.cache])           # (T, H)
        dlog = np.stack([c[7] for c in self.cache])         # (T, 256)  softmax outputs
        for t, c in enumerate(self.cache):
            dlog[t, c[8]] -= 1.0                            # softmax - onehot
        dlog /= T
        G["Wo"] = Hs.T @ dlog
        G["bo"] = dlog.sum(0)
        dH = dlog @ P["Wo"].T                               # (T, H) dL/dh_t from the outputs

        dh_next = np.zeros(H, dtype=self.dtype)
        dA = np.empty((T, 3 * H), dtype=self.dtype)         # dL/d(pre-activations)
        toks = np.empty(T, dtype=np.int64)
        Hprev = np.empty((T, H), dtype=self.dtype)
        RH = np.empty((T, H), dtype=self.dtype)
        for t in range(T - 1, -1, -1):
            tok_in, h_prev, z, r, rh, hc, h_new, p, b = self.cache[t]
            dh = dH[t] + dh_next
            dz = dh * (hc - h_prev)
            dhc = dh * z
            dh_prev = dh * (1.0 - z)
            da_h = dhc * (1.0 - hc * hc)
            drh = da_h @ Uh.T
            dh_prev = dh_prev + drh * r
            dr = drh * h_prev
            da_z = dz * z * (1.0 - z)
            da_r = dr * r * (1.0 - r)
            dh_prev = dh_prev + da_z @ Uz.T + da_r @ Ur.T
            dA[t, :H], dA[t, H:2 * H], dA[t, 2 * H:] = da_z, da_r, da_h
            toks[t], Hprev[t], RH[t] = tok_in, h_prev, rh
            dh_next = dh_prev

        X = P["E"][toks]                                    # (T, D)
        G["W"] = X.T @ dA
        G["b"] = dA.sum(0)
        G["U"][:, :2 * H] = Hprev.T @ dA[:, :2 * H]
        G["U"][:, 2 * H:] = RH.T @ dA[:, 2 * H:]
        np.add.at(G["E"], toks, dA @ P["W"].T)
        return G

    def _adam(self, G):
        norm = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
        if self.clip and norm > self.clip:
            scale = self.dtype(self.clip / norm)
            for g in G.values():
                g *= scale
        self.step += 1
        b1, b2 = self.beta1, self.beta2
        lr_t = self.lr * np.sqrt(1 - b2 ** self.step) / (1 - b1 ** self.step)
        if self.lr_linear_to is not None and self.total_steps:
            frac = min(1.0, self.step / self.total_steps)
            lr_t *= (1 - frac) + frac * (self.lr_linear_to / self.lr)
        elif self.lr_decay:
            lr_t /= np.sqrt(1.0 + self.step / self.lr_decay)
        for k, W in self.P.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * G[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * (G[k] * G[k])
            W -= (lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)).astype(self.dtype)

    # -- used by the gradient-check test ------------------------------------
    def window_loss(self, tokens, h0, prev=START):
        """Mean NLL over a window, pure forward pass, no side effects."""
        P, H = self.P, self.H
        h = h0
        loss = 0.0
        for b in tokens:
            x = P["E"][prev]
            a = x @ P["W"] + P["b"]
            hU = h @ P["U"]
            z = _sigmoid(a[:H] + hU[:H])
            r = _sigmoid(a[H:2 * H] + hU[H:2 * H])
            hc = np.tanh(a[2 * H:] + (r * h) @ P["U"][:, 2 * H:])
            h = (1.0 - z) * h + z * hc
            logits = h @ P["Wo"] + P["bo"]
            logits = logits - logits.max()
            loss -= float(logits[b] - np.log(np.exp(logits).sum()))
            prev = b
        return loss / len(tokens)


MODELS = {
    "gru": lambda: GRUByteModel(),
    "gru256": lambda: GRUByteModel(hidden=256, lr=1e-3),
}
