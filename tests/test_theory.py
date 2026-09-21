"""Numerical checks of the theoretical guarantees the README relies on."""
import math
import random

import numpy as np

from neuralzip.codec import encode, ideal_bits, quantize
from neuralzip.coder import PROB_TOTAL
from neuralzip.mixing import GeoMixture, Mixture
from neuralzip.models import ContextMix, Order0


class Fixed:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)
        self.p /= self.p.sum()

    def predict(self):
        return self.p

    def update(self, b):
        pass


def test_arithmetic_coder_meets_shannon_bound_with_explicit_overhead():
    """|code| <= sum(-log2 p) + per-symbol quantisation loss + 2 bits termination + byte padding.

    Quantisation: every symbol's integer frequency f satisfies
    f > p * (PROB_TOTAL - 256), so the coded probability is at least
    p * (1 - 256/PROB_TOTAL) and costs at most -log2(1 - 256/PROB_TOTAL) extra
    bits per symbol (about 0.0057).  Termination costs 2 bits (Witten, Neal &
    Cleary 1987) and the final byte is padded with at most 7 zero bits.
    """
    rng = np.random.default_rng(7)
    per_symbol = -math.log2(1 - 256 / PROB_TOTAL)
    for conc in (0.02, 0.2, 2.0):
        probs = rng.dirichlet(np.full(256, conc))
        data = bytes(random.Random(1).choices(range(256), weights=probs, k=40000))
        coded_bits = 8 * len(encode(data, Fixed(probs)))
        ideal = ideal_bits(data, Fixed(probs))
        assert coded_bits <= ideal + per_symbol * len(data) + 2 + 7
        # and it is not much better than ideal either: the coder is tight, not magic
        assert coded_bits >= ideal - 8


def test_bayesian_mixture_regret_is_at_most_log2_N_bits():
    """Bayesian mixture (gamma=1, beta=1, no floor): code length <= best expert + log2(N).

    Standard result for exponential weights under log-loss (Cesa-Bianchi &
    Lugosi 2006, ch. 9): the mixture's cumulative log loss is at most the best
    expert's plus ln(N), i.e. log2(N) bits.
    """
    data = open("corpora/english.txt", "rb").read()[:30000]
    experts = [Order0(), ContextMix(1), ContextMix(2), ContextMix(3)]
    n = len(experts)
    mix = Mixture(experts, beta=1.0, gamma=1.0, floor=0.0)
    mix_bits = 0.0
    expert_bits = np.zeros(n)
    for b in data:
        p = mix.predict()
        mix_bits -= math.log2(p[b])
        for i, pi in enumerate(mix._preds):
            expert_bits[i] -= math.log2(max(pi[b], 1e-300))
        mix.update(b)
    assert mix_bits <= expert_bits.min() + math.log2(n) + 1e-6
    # the bound is not vacuous: the best expert is much better than the worst
    assert expert_bits.max() > expert_bits.min() * 1.3


def test_geometric_mixer_gradient_matches_finite_differences():
    """d(-log p(b))/dw_i = E_p[log p_i] - log p_i(b), checked numerically."""
    rng = np.random.default_rng(3)
    experts = [Fixed(rng.dirichlet(np.full(256, 0.1))) for _ in range(3)]
    mix = GeoMixture(experts, lr=0.0)
    mix.W[0] = np.array([0.7, 0.2, 0.4])
    b = 65

    def loss(W):
        mix.W[0] = W
        return -math.log(mix.predict()[b])

    W0 = mix.W[0].copy()
    mix.predict()
    analytic = mix._logs @ mix._p - mix._logs[:, b]
    for i in range(3):
        e = np.zeros(3); e[i] = 1e-6
        num = (loss(W0 + e) - loss(W0 - e)) / 2e-6
        assert abs(num - analytic[i]) < 1e-6 * (1 + abs(num)), (i, num, analytic[i])


def test_geometric_mixture_can_be_sharper_than_every_expert():
    """With weights summing to more than 1, agreeing experts produce a sharper prediction.

    A linear mixture can never assign a symbol more probability than the most
    confident expert does; a geometric one with w = [1, 1] gives p ∝ p1 * p2.
    """
    p = np.full(256, 0.5 / 255); p[65] = 0.5
    experts = [Fixed(p), Fixed(p)]
    geo = GeoMixture(experts, lr=0.0, init=[1.0, 1.0])
    lin = Mixture(experts)
    assert geo.predict()[65] > 0.99
    assert abs(lin.predict()[65] - 0.5) < 1e-9


def test_ppm_escape_is_witten_bell_interpolation():
    """alpha=1 reproduces Witten-Bell smoothing exactly: lambda = N / (N + distinct)."""
    m = ContextMix(max_order=1, alpha=1.0)
    for b in b"abacabad":
        m.predict(); m.update(b)
    # context 'a' has seen b, c, b, d -> N=4, distinct=3
    p_lower = m.dense[0][b""] / m.dense[0][b""].sum()
    counts = m.dense[1][b"a"]
    expected = (counts + 3 * p_lower) / (4 + 3)
    m.hist = b"a"
    assert np.allclose(m.predict(), expected)
