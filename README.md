# neuralzip — lossless compression with a neural network that learns as it goes

[![tests](https://github.com/ayaan-cs/neuralzip/actions/workflows/ci.yml/badge.svg)](https://github.com/ayaan-cs/neuralzip/actions/workflows/ci.yml)

A from-scratch compressor (Python 3.12 + NumPy, no ML frameworks, nothing
pre-trained) that beats `gzip -9`, `bzip2` and `xz` on text by replacing the
*modelling* half of a compressor with a recurrent neural network trained
**online, during compression**. No weights are stored in the archive: the
decoder re-trains the same network on the bytes it has already decoded and
stays in lock-step with the encoder — the construction used by NNCP [[9]](#references)
and cmix [[10]](#references), scaled down to something readable in an afternoon.

```
python make_corpora.py                          # builds corpora/ from files already on the machine
python neuralzip.py compress   corpora/english.txt out.nz
python neuralzip.py decompress out.nz restored.txt
python neuralzip.py info       out.nz
python bench.py                                 # full table, every model round-trip verified
python -m pytest tests                          # coder, gradient checks, theory bounds, round trips
```

Requires `numpy` (and `pytest` for the tests). Nothing else.

## Results

<!-- RESULTS -->
| model | english.txt (517,537 B) | python.txt (293,103 B) | random.bin (32,768 B) |
|---|---:|---:|---:|
| gzip -9 | 2.251 bpb | 1.861 bpb | 8.004 bpb |
| bzip2 | 1.622 bpb (72% of gzip) | 1.569 bpb (84% of gzip) | 8.119 bpb (101% of gzip) |
| xz (lzma) | 1.579 bpb (70% of gzip) | 1.642 bpb (88% of gzip) | 8.015 bpb (100% of gzip) |
| order0 | 4.696 bpb (209% of gzip) | 4.356 bpb (234% of gzip) | 8.023 bpb (100% of gzip) |
| ctx4 | 1.739 bpb (77% of gzip) | 1.666 bpb (90% of gzip) | 8.420 bpb (105% of gzip) |
| ctx8 | 1.622 bpb (72% of gzip) | 1.616 bpb (87% of gzip) | 8.420 bpb (105% of gzip) |
| ctx | 1.430 bpb (64% of gzip) | 1.456 bpb (78% of gzip) | 8.136 bpb (102% of gzip) |
| gru | 1.935 bpb (86% of gzip) | 1.740 bpb (93% of gzip) | 8.045 bpb (101% of gzip) |
| nz | 1.319 bpb (59% of gzip) | 1.342 bpb (72% of gzip) | 8.002 bpb (100% of gzip) |

`bpb` = bits per byte of the original (8.0 = no compression). Every neuralzip
row is decoded and compared byte-for-byte before being reported. Corpora:
Python's own documentation topics (`pydoc_data.topics`) as English prose; six
stdlib modules concatenated as source code; 32 KB of `random.randbytes` as an
incompressible control.

For scale, on the standard `enwik8` benchmark (100 MB of Wikipedia) the same
reference compressors score gzip -9 2.92 bpb, xz -9 1.99 bpb, and the
state-of-the-art neural compressors cmix v18 1.19 bpb and NNCP v2 1.20 bpb
[[9]](#references); cmix v21 reaches 1.17 bpb [[10]](#references). Those systems
use models thousands of times larger, run for hours to days, and have 100–1000×
more data to learn from than the half-megabyte here. The point of this project
is the mechanism, not the leaderboard.

## The idea

Shannon's source coding theorem [[1]](#references) says the shortest achievable
expected code length for a source is its entropy, and for a specific sequence
the target is its information content, `Σ_t −log₂ p(x_t | x_<t)`. An
**arithmetic coder** [[2]](#references) achieves that to within 2 bits for the
whole message, for *any* distribution you hand it, and its cost is additive
over symbols [[3]](#references) — so the only remaining question is how good
your `p` is. Compression and next-byte prediction are the same problem; a
language model's cross-entropy on a file *is* its compressed size. This
equivalence is old [[4]](#references), [[5]](#references) and was recently
restated at LLM scale [[6]](#references).

```
                 ┌──────────────────┐
   bytes ──┬───▶ │ context model    │──┐   log-linear      ┌────────────┐
           │     │ (orders 0–24)    │  ├─▶ mixing ──▶ p(x) ─▶│ arithmetic │──▶ bits
           ├───▶ │ GRU, online      │──┘   (weights learned) │   coder    │
           │     └──────────────────┘                        └────────────┘
           └──────────── every model updates on the byte that actually arrived ──▶
```

### Pieces

| file | what it is |
|---|---|
| `neuralzip/coder.py` | 32-bit integer arithmetic coder after Witten, Neal & Cleary [[2]](#references): 16-bit frequencies, underflow ("pending bits") handling, 2-bit termination. |
| `neuralzip/codec.py` | Model ⇄ coder glue. Quantises float probabilities to an integer table with every symbol ≥ 1. |
| `neuralzip/models.py` | `Order0` (adaptive frequencies); `ContextMix`: PPM-style blending of context orders with a Witten–Bell / PPM-C escape estimate [[7]](#references), [[8]](#references), sparse packed storage for long contexts; `MatchModel`: follows the last occurrence of the current context, with confidence learned per match length (the PAQ-family match model [[15]](#references)). |
| `neuralzip/neural.py` | `GRUByteModel`: embed → GRU [[11]](#references) → softmax over 256 bytes. Hand-written forward *and* backward pass; truncated BPTT [[12]](#references) every 16 bytes; Adam [[13]](#references) with NNCP's settings; optional NNCP-v2-style periodic retraining. |
| `neuralzip/mixing.py` | `GeoMixture`: geometric (log-linear) mixing [[14]](#references) with weights learned by online gradient descent on code length, the multi-symbol form of PAQ's logistic mixing [[15]](#references). `Mixture`: Bayesian / fixed-share mixture [[16]](#references), [[17]](#references), kept for comparison. |
| `neuralzip.py` | CLI and container format (`NZ01`, model name, length, CRC-32, payload). |
| `bench.py`, `trace.py` | Benchmark table; JSON trace of per-byte predictions for the visualizer. |
| `experiments/` | The sweep scripts behind every number in this README. |

### Why online learning works for compression

The decoder has seen exactly the bytes the encoder had seen when it made each
prediction. So any deterministic function of the past — including "train a
neural network on it" — is available to both sides for free. Bellard states
the same requirement for NNCP: "the decoder works symmetrically so there is no
need to transmit the model parameters. It implies both encoder and decoder
update their model identically" [[9]](#references). The constraints that
follow shape the code:

* **Determinism.** The GRU runs in float32 with BLAS pinned to one thread
  (`neuralzip/__init__.py`), so encoder and decoder execute bit-identical
  floating-point operations, and probabilities are quantised to integers
  *before* they touch the coder. NNCP has the same constraint and the same
  caveat: identical results are "guaranteed only if the code is running with
  the exact same hardware and software versions" [[9]](#references).
* **No symbol may have probability 0.** `quantize()` gives every byte at least
  frequency 1 of 65 536; otherwise a single surprising byte would be uncodable.
  The cost is at most `−log₂(1 − 256/65536) ≈ 0.0057` bits per byte, and
  `tests/test_theory.py` checks the coder's output against
  `Σ −log₂ p + 0.0057·n + 2 + 7` (quantisation, termination, byte padding).
* **Single pass, non-stationary.** Every byte is seen once, so there is no
  overfitting to guard against — NNCP uses no dropout for the same reason
  [[9]](#references) — but the optimiser must track a moving target. Adam with
  β₁ = 0 (i.e. bias-corrected RMSProp, the setting NNCP uses [[9]](#references))
  beat the textbook β₁ = 0.9 by 4–5% in every sweep, and a decaying learning
  rate `lr / √(1 + step/1500)` beat a constant one.

### The context model is Witten–Bell smoothing

`ContextMix` blends orders recursively:

```
p_k(x | ctx_k) = ( n_k(x) + e_k · p_{k−1}(x) ) / ( N_k + e_k ),     e_k = α · d_k
```

where `d_k` is the number of distinct symbols seen after this context. With
α = 1 this is exactly the PPM-C escape estimate [[7]](#references) — in
language-modelling terms, interpolated Witten–Bell smoothing [[8]](#references)
(`tests/test_theory.py` checks the equivalence). On this data a *larger*
escape, α = 6, is markedly better (1.53 vs 1.66 bpb at order 8): byte-level
contexts with one or two observations are less trustworthy than the raw count
suggests. Interpolated absolute discounting (D = 0.5…0.9) was worse
(1.85 bpb); Chen & Goodman [[8]](#references) attribute Kneser–Ney's advantage
to its modified lower-order *continuation counts*, which are not implemented
here.

The orders modelled are 0–6 and then 8, 12, 16, 24. Long contexts keep
helping because documentation repeats long strings — the PPM* observation
that unbounded contexts are usually deterministic [[18]](#references). To
make that affordable, a context that has only ever been followed by one
symbol is stored as a single packed integer; a dict is allocated only once a
second symbol appears. That halves memory for identical predictions (876 →
368 MB for dense orders 0–16 on the 517 KB corpus), and the sparse ladder then
gets to 201 MB with *better* predictions than dense order 16 (1.435 vs 1.451 bpb).

### Why geometric mixing beats "pick the best expert"

A Bayesian mixture `p = Σ w_i p_i` with posterior weights is guaranteed to be
within `log₂ N` bits of the best single expert over the whole file
[[16]](#references) — `tests/test_theory.py` checks this bound numerically —
but it can never be *better* than the best expert, and in practice its weights
collapse onto one model. Log-linear (geometric) mixing [[14]](#references)

```
p(x) ∝ Π_i p_i(x)^{w_i},        ∂(−log p(x))/∂w_i = E_p[log p_i] − log p_i(x)
```

has unconstrained weights: when two partly independent experts agree, the
mixture becomes sharper than either (checked in the tests), and Mattern shows
the weight optimisation is convex and that this is the rationale behind PAQ's
logistic mixing [[14]](#references), [[15]](#references). On the English
corpus the context model alone gets 1.43 bpb and the GRU alone 1.94, but the
mixture gets 1.32. The learned weights also expose structure: an order-8 model
added beside the deeper ladder receives a *negative* weight — the mixer uses
it as a correction term, not a vote.

### What did not help (and is still in the code as an option)

* **Replaying the current 16-byte window** for extra gradient steps: worse.
* **NNCP-v2-style periodic retraining** [[19]](#references) — every 50 KB, one
  extra pass over the last 50 KB at half the learning rate: the GRU alone
  improves (2.33 → 2.27 bpb on 200 KB), but inside the mixture the gain
  vanishes (1.322 vs 1.319) while costing 1.6× the time. Available as
  `--model nz-retrain`. Bellard reports large gains from retraining, but with
  a Transformer, tens of megabytes of history and many "epochs"; at half a
  megabyte there is little to re-learn.
* **Bigger GRU** (192 or 256 units): no better at this data size; the
  learning curve, not capacity, is the bottleneck.

## What the numbers mean

* `gzip` is DEFLATE [[20]](#references): LZ77 string matching plus Huffman
  coding. It cannot exploit "after `the ` a vowel is likely" unless that exact
  string repeated recently. `xz` is LZMA (LZ77 with range coding);
  `bzip2` is the Burrows–Wheeler transform plus entropy coding [[21]](#references).
* `ctx` is essentially PPM (the family behind `7z`'s PPMd) and beats all three
  by itself.
* `gru` alone loses to `ctx` on half a megabyte — a network trained from random
  init in one pass is data-hungry — but its per-segment learning curve is still
  falling at the end of the file while the context model's has flattened.
* `nz` is the mixture. On random bytes it costs nothing: the context models
  alone are 2–5% *worse* than raw (overconfident after one observation), but
  the mixer learns within a few hundred bytes to give them no weight.

## Visualizer

`python trace.py` replays the headline model over the English corpus and
writes `trace.json`: the per-segment learning curve of every expert plus, for a
1.2 KB window, what each predictor expected next, the mixer weights and the
bits paid for every byte. That file drives a three-board Claude Design canvas
(live byte-by-byte replay, learning curve, pipeline diagram):
https://claude.ai/artifact/YbhHTJnx1YDDmvtYYfgN6g — the artboard sources are
in `visualizer/`.

## Limits, honestly

* **Speed:** ~9 KB/s. The GRU forward pass is a handful of 128×384
  matrix-vector products per byte in NumPy. NNCP's LSTM in a custom C library
  ran at ~1 KB/s on *enwik9* with a model ~1000× larger [[9]](#references); cmix
  needs 32 GB of RAM and 18 hours for enwik8 [[10]](#references). Neural
  compression is slow by construction — every byte is a training step.
* **Memory:** the context tables grow with the input (~200 MB on 517 KB).
  Hashed, bounded tables (as in PAQ) would fix this at a small cost in ratio.
* **Portability of archives:** an archive decodes correctly on the same
  NumPy/BLAS build. Different BLAS libraries can round differently, which would
  desynchronise the decoder — the same caveat NNCP carries [[9]](#references).
  A production system would use integer or otherwise deterministic arithmetic.
* **Data size:** the neural expert only starts to pay off after ~100 KB.
  Everything about this design gets better with more data; nothing about it
  gets worse.

## Tests

* `tests/test_coder.py` — the arithmetic coder round-trips uniform, skewed,
  adaptive and degenerate inputs and lands within 0.5% of the Shannon bound.
* `tests/test_neural.py` — the hand-written BPTT gradient matches central
  finite differences on every parameter tensor; the GRU actually learns (a
  repeating pattern ends up costing < 0.2 bits/byte); round trips.
* `tests/test_theory.py` — the guarantees the README relies on, checked
  numerically: the coder's explicit overhead bound, the Bayesian mixture's
  `log₂ N` regret bound, the geometric mixer's gradient, its ability to sharpen,
  and the Witten–Bell equivalence.
* `tests/test_match.py` — the match model predicts only from a byte-for-byte
  verified context (never from a hash collision), abstains with the uniform
  distribution when it has no match, learns to trust long matches more than
  short ones, and — because a constant log-vector cancels in a softmax — moves
  a geometric mixture by exactly nothing while it is silent.

GitHub Actions runs the suite plus a full CLI round trip on 3.11, 3.12 and 3.13
on every push (`.github/workflows/ci.yml`).

## References

1. C. E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal* 27, 1948.
2. I. H. Witten, R. M. Neal and J. G. Cleary, "Arithmetic Coding for Data Compression," *Communications of the ACM* 30(6):520–540, 1987. https://dl.acm.org/doi/10.1145/214762.214771
3. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006 — ch. 5 (source coding, Kraft inequality) and §13.3 (arithmetic coding). See also Y. Yang, S. Mandt and L. Theis, "An Introduction to Neural Data Compression," 2022, §2 for the "information content + ~2 bits" statement. https://arxiv.org/abs/2202.06533
4. J. Schmidhuber and S. Heil, "Sequential Neural Text Compression," *IEEE Transactions on Neural Networks* 7(1):142–146, 1996. https://people.idsia.ch/~juergen/ieeetnn1996.pdf
5. M. V. Mahoney, "Fast Text Compression with Neural Networks," *FLAIRS 2000*. https://cs.fit.edu/~mmahoney/compression/mmahoney00.pdf
6. G. Delétang, A. Ruoss, P.-A. Duquenne, E. Catt, T. Genewein, C. Mattern, J. Grau-Moya, L. K. Wenliang, M. Aitchison, L. Orseau, M. Hutter and J. Veness, "Language Modeling Is Compression," *ICLR 2024*. https://arxiv.org/abs/2309.10668
7. A. Moffat, "Implementing the PPM Data Compression Scheme," *IEEE Transactions on Communications* 38(11):1917–1921, 1990 (method C: escape count = number of distinct symbols). https://www.semanticscholar.org/paper/0f0b4d4fdec3c7a9c4cb06ca5b78254f281bbf59
8. S. F. Chen and J. Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling," *Computer Speech & Language* 13(4):359–394, 1999 (Witten–Bell and Kneser–Ney; interpolated KN best). https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf
9. F. Bellard, "Lossless Data Compression with Neural Networks," 2019 (https://bellard.org/nncp/nncp.pdf) and "NNCP v2: Lossless Data Compression with Transformer," 2021 (https://bellard.org/nncp/nncp_v2.1.pdf). Quotes and enwik8/enwik9 tables are from these; project page https://bellard.org/nncp/.
10. B. Knoll, *cmix*. https://www.byronknoll.com/cmix.html (v21: enwik8 → 14,623,723 bytes; mixing via a PAQ-style gated linear network plus an LSTM mixer).
11. K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk and Y. Bengio, "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," *EMNLP 2014* (the GRU). https://arxiv.org/abs/1406.1078
12. R. J. Williams and J. Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories," *Neural Computation* 2(4):490–501, 1990 (truncated BPTT).
13. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR 2015*. https://arxiv.org/abs/1412.6980 — gradient-norm clipping follows R. Pascanu, T. Mikolov and Y. Bengio, "On the difficulty of training recurrent neural networks," *ICML 2013*. https://arxiv.org/abs/1211.5063
14. C. Mattern, "Mixing Strategies in Data Compression," *Data Compression Conference 2012*. https://arxiv.org/abs/1302.2839 (geometric weighting; "superior to linear weighting"; the rationale behind PAQ weighting).
15. M. V. Mahoney, "Adaptive Weighing of Context Models for Lossless Data Compression," Florida Tech technical report CS-2005-16, 2005. https://mattmahoney.net/dc/cs200516.pdf — and *Data Compression Explained*, https://mattmahoney.net/dc/dce.html.
16. N. Cesa-Bianchi and G. Lugosi, *Prediction, Learning, and Games*, Cambridge University Press, 2006 — ch. 9 (logarithmic loss: the exponentially-weighted / Bayesian mixture is within `ln N` of the best expert).
17. M. Herbster and M. K. Warmuth, "Tracking the Best Expert," *Machine Learning* 32:151–178, 1998 (fixed share).
18. J. G. Cleary and W. J. Teahan, "Unbounded Length Contexts for PPM," *The Computer Journal* 40(2/3):67–75, 1997 (PPM*: deterministic long contexts).
19. F. Bellard, NNCP v2 [[9]](#references), §2.3: "we retrain the model using the already decompressed data at regular intervals."
20. P. Deutsch, "DEFLATE Compressed Data Format Specification version 1.3," RFC 1951, 1996. J. Ziv and A. Lempel, "A Universal Algorithm for Sequential Data Compression," *IEEE Transactions on Information Theory* 23(3):337–343, 1977.
21. M. Burrows and D. J. Wheeler, "A Block-sorting Lossless Data Compression Algorithm," DEC SRC Research Report 124, 1994.
