"""The match model: predict what followed this context the last time it appeared."""
import numpy as np

from neuralzip.codec import decode, encode, ideal_bits
from neuralzip.mixing import GeoMixture
from neuralzip.models import MatchModel


class Fixed:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)
        self.p /= self.p.sum()

    def predict(self):
        return self.p

    def update(self, b):
        pass


def test_a_long_repeat_becomes_nearly_free():
    data = b"the quick brown fox jumps over the lazy dog. " * 40
    assert ideal_bits(data, MatchModel()) / len(data) < 0.5


def test_it_abstains_when_it_has_no_match():
    """No match means the uniform distribution -- 8 bits, and no opinion."""
    m = MatchModel(min_len=6)
    assert np.allclose(m.predict(), 1 / 256)
    for b in b"abcdefghij":  # every 6-byte context here is unique
        m.predict()
        m.update(b)
    assert np.allclose(m.predict(), 1 / 256)


def test_a_silent_match_model_does_not_move_a_geometric_mixture():
    """A constant log-vector cancels in the softmax, whatever weight it carries.

    This is why the match model is safe to add: when it has nothing to say it
    costs the mixture exactly nothing.
    """
    p = np.random.default_rng(5).dirichlet(np.full(256, 0.3))
    solo = GeoMixture([Fixed(p)], lr=0.0, init=[1.0])
    duo = GeoMixture([Fixed(p), MatchModel()], lr=0.0, init=[1.0, 0.7])
    assert np.allclose(solo.predict(), duo.predict())


def test_the_predicted_byte_always_follows_a_verified_context():
    """Never predict from a hash collision: the context bytes are compared.

    Invariant: whenever a match is live, the `min_len` bytes before the pointer
    are exactly the last `min_len` bytes seen.  It holds when the match is
    found (the bytes are compared) and is preserved by every extension.
    """
    data = open("corpora/english.txt", "rb").read()[:20000]
    m = MatchModel(min_len=6)
    live = 0
    for b in data:
        m.predict()
        if m.mlen:
            k = m.min_len
            assert bytes(m.hist[m.ptr - k:m.ptr]) == bytes(m.hist[-k:])
            live += 1
        m.update(b)
    assert live > 1000  # and it is live often enough for the check to mean something


def test_confidence_is_learned_not_assumed():
    """Long matches must end up trusted more than short ones."""
    m = MatchModel(min_len=6)
    for b in open("corpora/english.txt", "rb").read()[:30000]:
        m.predict()
        m.update(b)
    rate = m.hits / (m.hits + m.miss)
    assert rate[m.MAXLEN] > rate[m.min_len] > 0.5


def test_roundtrip():
    data = open("corpora/python.txt", "rb").read()[:8000]
    out = encode(data, MatchModel())
    assert decode(out, len(data), MatchModel()) == data
