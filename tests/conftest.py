"""Import neuralzip before anything pulls in numpy.

The package pins BLAS to a single thread (neuralzip/__init__.py), but BLAS only
reads those environment variables when numpy first loads.  pytest imports this
file before any test module, so the pin is in place for the whole suite.
"""
import neuralzip  # noqa: F401
