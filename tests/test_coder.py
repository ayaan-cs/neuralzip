import math
import random

import numpy as np

from neuralzip.codec import decode, encode, ideal_bits, quantize
from neuralzip.coder import PROB_TOTAL


class FixedModel:
    """Static distribution; lets us check code length against the entropy bound."""

    def __init__(self, probs):
        self.p = np.asarray(probs, dtype=np.float64)
        self.p /= self.p.sum()

    def predict(self):
        return self.p

    def update(self, b):
        pass


def _sample(probs, n, seed):
    rng = random.Random(seed)
    return bytes(rng.choices(range(256), weights=probs, k=n))


def test_quantize_invariants():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet(np.full(256, 0.05))
        cum = quantize(p)
        f = np.diff(cum)
        assert cum[0] == 0 and cum[-1] == PROB_TOTAL
        assert f.min() >= 1
    # very peaked distribution still leaves room for every symbol
    p = np.full(256, 1e-9); p[65] = 1.0
    cum = quantize(p / p.sum())
    assert np.diff(cum).min() >= 1 and cum[-1] == PROB_TOTAL


def test_roundtrip_uniform():
    data = bytes(random.Random(1).randrange(256) for _ in range(20000))
    m = FixedModel(np.ones(256))
    out = encode(data, m)
    assert decode(out, len(data), FixedModel(np.ones(256))) == data
    # uniform => 8 bits/byte, no better and barely no worse
    assert abs(len(out) - len(data)) <= 4


def test_roundtrip_skewed_near_entropy():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.full(256, 0.02))
    data = _sample(probs, 50000, seed=3)
    m = FixedModel(probs)
    out = encode(data, m)
    assert decode(out, len(data), FixedModel(probs)) == data
    ideal = ideal_bits(data, FixedModel(probs))
    # quantisation to 16-bit frequencies costs a tiny amount; coder termination costs 2 bits
    assert len(out) * 8 <= ideal * 1.005 + 16


def test_roundtrip_edge_cases():
    for data in (b"", b"a", b"\x00" * 1000, b"\xff" * 1000, bytes(range(256)) * 10):
        probs = np.ones(256)
        out = encode(data, FixedModel(probs))
        assert decode(out, len(data), FixedModel(probs)) == data


def test_roundtrip_adaptive_model_in_lockstep():
    """Model whose predictions depend on history: decoder must mirror encoder exactly."""

    class Adaptive:
        def __init__(self):
            self.counts = np.ones(256)

        def predict(self):
            return self.counts / self.counts.sum()

        def update(self, b):
            self.counts[b] += 1

    data = ("the quick brown fox jumps over the lazy dog " * 500).encode()
    out = encode(data, Adaptive())
    assert decode(out, len(data), Adaptive()) == data
    assert len(out) < len(data) * 0.7
