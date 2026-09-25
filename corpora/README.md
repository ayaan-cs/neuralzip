# Benchmark corpora

These files are *inputs* to the benchmark, not part of this project's source,
and two of the three are **not** covered by the repository's MIT licence.

| file | what it is | licence |
|---|---|---|
| `english.txt` | The Python help topics (`pydoc_data.topics`), concatenated as plain prose | PSF License Agreement |
| `python.txt` | Six CPython standard library modules concatenated: `argparse`, `textwrap`, `json/encoder`, `json/decoder`, `dataclasses`, `enum` | PSF License Agreement |
| `random.bin` | 32 KB of `random.randbytes` with seed 0 | MIT, with the rest of this repo |

`english.txt` and `python.txt` are verbatim extracts from CPython.
Copyright © 2001–2026 Python Software Foundation; All Rights Reserved.
They are redistributed here under the PSF License Agreement
(https://docs.python.org/3/license.html), which permits redistribution provided
this notice is retained.

They are committed rather than downloaded so that every number in the README is
reproducible against the exact bytes that produced it. `python make_corpora.py`
rebuilds them from the CPython installation on the current machine, which on a
different Python version yields *different* files and therefore different
bits-per-byte figures.
