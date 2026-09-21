#!/usr/bin/env python
"""neuralzip -- compress files with a neural network that learns as it goes.

    python neuralzip.py compress   input.txt  output.nz  [--model nz]
    python neuralzip.py decompress output.nz  restored.txt
    python neuralzip.py info       output.nz

Container format (little-endian):
    magic  b"NZ01"
    u8     model-name length, then the name (ASCII)
    u64    original length in bytes
    u32    CRC-32 of the original data
    ...    arithmetic-coded payload

No model weights are stored: the decoder rebuilds them by training on the
data it has already decoded, in lock-step with the encoder.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib

from neuralzip.codec import decode, encode
from neuralzip.registry import MODELS

MAGIC = b"NZ01"


def pack(model_name: str, data: bytes, payload: bytes) -> bytes:
    name = model_name.encode("ascii")
    return MAGIC + struct.pack("<B", len(name)) + name + struct.pack("<QI", len(data), zlib.crc32(data)) + payload


def unpack(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("not a neuralzip file")
    n = blob[4]
    name = blob[5:5 + n].decode("ascii")
    length, crc = struct.unpack_from("<QI", blob, 5 + n)
    return name, length, crc, blob[5 + n + 12:]


class _Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0 = total, label, time.perf_counter()

    def __call__(self, i):
        dt = time.perf_counter() - self.t0
        rate = i / dt / 1024 if dt > 0 else 0.0
        print(f"\r{self.label}: {i:,}/{self.total:,} bytes  {rate:5.1f} KB/s", end="", file=sys.stderr, flush=True)

    def done(self):
        print(f"\r{self.label}: {self.total:,}/{self.total:,} bytes  {time.perf_counter() - self.t0:.1f}s", file=sys.stderr)


def cmd_compress(a):
    data = open(a.input, "rb").read()
    if a.model not in MODELS:
        sys.exit(f"unknown model {a.model!r}; choose from {', '.join(MODELS)}")
    prog = _Progress(len(data), "compress")
    payload = encode(data, MODELS[a.model](), progress=prog)
    prog.done()
    blob = pack(a.model, data, payload)
    open(a.output, "wb").write(blob)
    bpb = 8 * len(blob) / max(len(data), 1)
    print(f"{a.input}: {len(data):,} -> {len(blob):,} bytes  ({bpb:.3f} bits/byte, "
          f"{len(data) / max(len(blob), 1):.2f}x, gzip -9 would be {len(zlib.compress(data, 9)):,})")


def cmd_decompress(a):
    name, length, crc, payload = unpack(open(a.input, "rb").read())
    prog = _Progress(length, "decompress")
    data = decode(payload, length, MODELS[name](), progress=prog)
    prog.done()
    if zlib.crc32(data) != crc:
        sys.exit("CRC mismatch: decoded data is corrupt")
    open(a.output, "wb").write(data)
    print(f"{a.input}: restored {length:,} bytes with model {name!r}, CRC ok")


def cmd_info(a):
    blob = open(a.input, "rb").read()
    name, length, crc, payload = unpack(blob)
    print(f"model: {name}\noriginal: {length:,} bytes\ncompressed: {len(blob):,} bytes\n"
          f"bits/byte: {8 * len(blob) / max(length, 1):.3f}\ncrc32: {crc:08x}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--model", default="nz", help=f"one of: {', '.join(MODELS)}")
    c.set_defaults(fn=cmd_compress)
    d = sub.add_parser("decompress")
    d.add_argument("input")
    d.add_argument("output")
    d.set_defaults(fn=cmd_decompress)
    i = sub.add_parser("info")
    i.add_argument("input")
    i.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
