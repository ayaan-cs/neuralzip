"""Build the benchmark corpora from files already on this machine (no downloads)."""
import os
import random
import sys

import pydoc_data.topics

os.makedirs("corpora", exist_ok=True)

# English prose: the Python language reference / help topics, as plain text.
text = "\n\n".join(pydoc_data.topics.topics[k] for k in sorted(pydoc_data.topics.topics))
with open("corpora/english.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

# Source code: a few stdlib modules concatenated.
lib = os.path.dirname(os.__file__)
parts = []
for name in ("argparse.py", "textwrap.py", "json/encoder.py", "json/decoder.py", "dataclasses.py", "enum.py"):
    with open(os.path.join(lib, name), encoding="utf-8") as f:
        parts.append(f.read())
with open("corpora/python.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(parts))

# Incompressible control: any honest compressor should come out at ~8 bits/byte.
random.seed(0)
with open("corpora/random.bin", "wb") as f:
    f.write(random.randbytes(32 * 1024))

for fn in sorted(os.listdir("corpora")):
    print(f"{fn:14s} {os.path.getsize(os.path.join('corpora', fn)):>8,d} bytes")
