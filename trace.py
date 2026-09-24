"""Export a JSON trace of the headline model for the visualiser.

Runs the mixture over a corpus once and records:
  * a learning curve: bits/byte per segment for each expert and the mixture
  * a detailed window: for every byte in [start, start+length), the top
    predictions of each expert and the mixture, the probability each assigned
    to the byte that actually came, the mixer weights and the bits paid.
"""
from __future__ import annotations

import argparse
import json
import zlib

from neuralzip.registry import MODELS  # first: pins BLAS to one thread before numpy loads

import numpy as np  # noqa: E402

TOPK = 6


def top(p, k=TOPK):
    idx = np.argsort(p)[::-1][:k]
    return [[int(i), round(float(p[i]), 4)] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpora/english.txt")
    ap.add_argument("--model", default="nz")
    ap.add_argument("--start", type=int, default=150_000)
    ap.add_argument("--length", type=int, default=1200)
    ap.add_argument("--segment", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="trace.json")
    a = ap.parse_args()

    data = open(a.corpus, "rb").read()
    if a.limit:
        data = data[:a.limit]
    mix = MODELS[a.model]()
    experts = mix.experts
    names = [getattr(e, "name", type(e).__name__) for e in experts]

    n_ex = len(experts)
    seg_bits = np.zeros(n_ex + 1)
    curve = {n: [] for n in names + ["mix"]}
    total_bits = np.zeros(n_ex + 1)
    window = []

    for i, b in enumerate(data):
        p_mix = mix.predict()
        p_ex = [np.exp(l) for l in mix._logs]  # experts' distributions, as the mixer saw them
        bits = [-np.log2(max(p[b], 1e-12)) for p in p_ex] + [-np.log2(max(p_mix[b], 1e-12))]
        seg_bits += bits
        total_bits += bits
        if a.start <= i < a.start + a.length:
            window.append({
                "b": b,
                "bits": round(float(bits[-1]), 3),
                "pa": [round(float(p[b]), 5) for p in p_ex] + [round(float(p_mix[b]), 5)],
                "w": [round(float(x), 3) for x in mix.weights()],
                "top": [top(p) for p in p_ex] + [top(p_mix)],
            })
        mix.update(b)
        if (i + 1) % a.segment == 0:
            for j, n in enumerate(names + ["mix"]):
                curve[n].append(round(float(seg_bits[j] / a.segment), 4))
            seg_bits[:] = 0
            print(f"\r{i + 1:,}/{len(data):,}", end="", flush=True)
    print()

    gz = len(zlib.compress(data, 9)) * 8 / len(data)
    out = {
        "corpus": a.corpus,
        "n": len(data),
        "experts": names,
        "segment": a.segment,
        "curve": curve,
        "overall": {n: round(float(total_bits[j] / len(data)), 4) for j, n in enumerate(names + ["mix"])},
        "gzip_bpb": round(gz, 4),
        "window_start": a.start,
        "window": window,
    }
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("overall bpb:", out["overall"], "gzip:", out["gzip_bpb"], "->", a.out)


if __name__ == "__main__":
    main()
