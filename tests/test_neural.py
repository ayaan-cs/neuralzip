import numpy as np

from neuralzip.codec import decode, encode
from neuralzip.neural import GRUByteModel, START


def test_gradients_match_finite_differences():
    """Analytic truncated-BPTT gradient vs central differences (float64, tiny model)."""
    m = GRUByteModel(hidden=6, embed=4, bptt=100, seed=3, dtype=np.float64)
    tokens = [104, 101, 108, 108, 111, 32, 119]
    h0 = (np.random.default_rng(1).standard_normal(6) * 0.3).astype(np.float64)
    m.h = h0.copy()
    for b in tokens:
        m.predict()
        m.update(b)  # bptt=100 so no training fires; cache holds the window
    G = m._gradients()

    eps = 1e-5
    rng = np.random.default_rng(0)
    for k, W in m.P.items():
        flat = W.reshape(-1)
        for i in rng.choice(flat.size, size=min(15, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = m.window_loss(tokens, h0)
            flat[i] = old - eps
            lm = m.window_loss(tokens, h0)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = G[k].reshape(-1)[i]
            assert abs(num - ana) < 1e-7 + 1e-5 * abs(num), (k, i, num, ana)


def test_roundtrip_small():
    data = ("def f(x):\n    return x * 2\n\n" * 60).encode()
    mk = lambda: GRUByteModel(hidden=32, embed=16, bptt=16)
    out = encode(data, mk())
    assert decode(out, len(data), mk()) == data
    assert len(out) < len(data) * 0.9


def test_learns_repeating_pattern():
    data = b"abcdefgh" * 2500
    out = encode(data, GRUByteModel(hidden=32, embed=16, bptt=16))
    assert len(out) * 8 / len(data) < 1.0  # trivially predictable => well under 1 bpb
    # and the *last* part must be nearly free: the model has learned the cycle
    m = GRUByteModel(hidden=32, embed=16, bptt=16)
    for b in data[:-800]:
        m.predict(); m.update(b)
    tail = 0.0
    for b in data[-800:]:
        tail -= float(np.log2(m.predict()[b])); m.update(b)
    assert tail / 800 < 0.2
