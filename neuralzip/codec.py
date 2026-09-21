"""Glue between a probability model and the arithmetic coder.

A model is any object with:
    predict() -> np.ndarray[256] float   probabilities for the next byte
    update(byte: int)                     observe the actual byte (learn / shift context)

The encoder and decoder drive the model in lock-step, so the model can be
anything deterministic -- including one that trains itself as it goes.
"""
from __future__ import annotations

import numpy as np

from .coder import ArithmeticDecoder, ArithmeticEncoder, PROB_TOTAL

_FLOOR_SCALE = PROB_TOTAL - 256


def quantize(probs: np.ndarray) -> np.ndarray:
    """Turn float probabilities into a cumulative integer table of 257 entries.

    Every symbol gets frequency >= 1 (so it stays codable) and the total is
    exactly PROB_TOTAL.  Deterministic for identical input, which is all the
    decoder needs.
    """
    p = np.asarray(probs, dtype=np.float64)
    f = np.floor(p * _FLOOR_SCALE).astype(np.int64) + 1
    f[int(np.argmax(p))] += PROB_TOTAL - int(f.sum())
    cum = np.empty(257, dtype=np.int64)
    cum[0] = 0
    np.cumsum(f, out=cum[1:])
    return cum


def encode(data: bytes, model, progress=None) -> bytes:
    enc = ArithmeticEncoder()
    for i, b in enumerate(data):
        cum = quantize(model.predict())
        enc.encode(int(cum[b]), int(cum[b + 1]))
        model.update(b)
        if progress and i % 8192 == 0:
            progress(i)
    return enc.finish()


def decode(payload: bytes, n: int, model, progress=None) -> bytes:
    dec = ArithmeticDecoder(payload)
    out = bytearray(n)
    for i in range(n):
        cum = quantize(model.predict())
        t = dec.target()
        b = int(np.searchsorted(cum, t, side="right")) - 1
        dec.consume(int(cum[b]), int(cum[b + 1]))
        out[i] = b
        model.update(b)
        if progress and i % 8192 == 0:
            progress(i)
    return bytes(out)


def ideal_bits(data: bytes, model) -> float:
    """Shannon code length sum(-log2 p) -- what a perfect coder would achieve."""
    total = 0.0
    for b in data:
        p = model.predict()
        total -= float(np.log2(max(p[b], 1e-12)))
        model.update(b)
    return total
