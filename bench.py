"""Benchmark every model against the stdlib compressors on every corpus.

Each neuralzip model is encoded, decoded, and verified byte-for-byte.
Reports bits per byte (bpb): the only number that matters -- 8.0 means no
compression, and the model's cross-entropy on the data is the floor.
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import os
import sys
import time
import zlib

from neuralzip.codec import decode, encode
from neuralzip.registry import MODELS


def stdlib_refs(data: bytes):
    yield "gzip -9", len(zlib.compress(data, 9))
    yield "bzip2", len(bz2.compress(data, 9))
    yield "xz (lzma)", len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


def run(models, corpora, limit, verify=True):
    rows = []
    for path in corpora:
        data = open(path, "rb").read()
        if limit:
            data = data[:limit]
        name = os.path.basename(path)
        for ref, size in stdlib_refs(data):
            rows.append(dict(corpus=name, n=len(data), model=ref, size=size, bpb=8 * size / len(data), enc_s=0.0, ok=True))
        for m in models:
            t0 = time.perf_counter()
            out = encode(data, MODELS[m]())
            t1 = time.perf_counter()
            ok = True
            if verify:
                ok = decode(out, len(data), MODELS[m]()) == data
            rows.append(dict(corpus=name, n=len(data), model=m, size=len(out), bpb=8 * len(out) / len(data), enc_s=t1 - t0, ok=ok))
            print(f"  {name:12s} {m:10s} {8*len(out)/len(data):6.3f} bpb  {t1-t0:7.1f}s  {'OK' if ok else 'FAIL'}", file=sys.stderr, flush=True)
    return rows


def markdown(rows):
    corpora = sorted({r["corpus"] for r in rows}, key=lambda c: [r["corpus"] for r in rows].index(c))
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    lines = ["| model | " + " | ".join(f"{c} ({next(r['n'] for r in rows if r['corpus']==c):,} B)" for c in corpora) + " |",
             "|---|" + "---:|" * len(corpora)]
    gz = {c: next(r["bpb"] for r in rows if r["corpus"] == c and r["model"] == "gzip -9") for c in corpora}
    for m in models:
        cells = []
        for c in corpora:
            r = next((r for r in rows if r["corpus"] == c and r["model"] == m), None)
            if r is None:
                cells.append("")
            else:
                s = f"{r['bpb']:.3f} bpb"
                if m != "gzip -9":
                    s += f" ({r['bpb']/gz[c]*100:.0f}% of gzip)"
                if not r["ok"]:
                    s += " **ROUNDTRIP FAIL**"
                cells.append(s)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--corpora", default="corpora/english.txt,corpora/python.txt,corpora/random.bin")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N bytes of each corpus")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = run(a.models.split(","), a.corpora.split(","), a.limit, verify=not a.no_verify)
    print()
    print(markdown(rows))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
