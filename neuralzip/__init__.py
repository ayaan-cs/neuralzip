"""neuralzip: lossless compression with a neural predictor + arithmetic coding.

The core idea: an arithmetic coder turns a probability model into a code whose
length is -log2 p(x) bits per symbol.  So a better predictor *is* a better
compressor.  Every model here exposes the same interface (see models.py) and is
run identically on the encoder and decoder side, so nothing but the coded bits
has to be transmitted.
"""
import os
import sys

# Neural models must produce bit-identical probabilities on both sides of the
# channel.  Multithreaded BLAS can change summation order between runs, so pin
# it before numpy is first imported anywhere in this package.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# BLAS reads those variables when it is loaded, which happens on the first
# `import numpy` in the process.  If numpy got there first the pin silently did
# nothing, so say so rather than quietly producing machine-dependent archives.
if "numpy" in sys.modules:  # pragma: no cover - import-order guard
    import warnings

    warnings.warn(
        "numpy was imported before neuralzip, so BLAS thread pinning did not take "
        "effect; results may not be reproducible on a machine with a different "
        "core count. Import neuralzip (or one of its submodules) first.",
        RuntimeWarning,
        stacklevel=2,
    )

from .coder import ArithmeticEncoder, ArithmeticDecoder, PROB_BITS, PROB_TOTAL  # noqa: E402

__all__ = ["ArithmeticEncoder", "ArithmeticDecoder", "PROB_BITS", "PROB_TOTAL"]
