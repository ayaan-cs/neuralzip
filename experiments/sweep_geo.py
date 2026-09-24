import sys, time
import neuralzip  # noqa: F401  -- before numpy: pins BLAS to one thread
import numpy as np
from neuralzip.codec import ideal_bits
from neuralzip.neural import GRUByteModel
from neuralzip.models import ContextMix
from neuralzip.mixing import GeoMixture, byte_class
data = open("corpora/english.txt","rb").read()[:int(sys.argv[1])]
gru_kw = eval("dict("+sys.argv[2]+")")
for s in sys.argv[3:]:
    kw = eval("dict("+s+")")
    if kw.pop("ctx", False):
        kw.update(n_ctx=4, ctx_fn=byte_class)
    t=time.perf_counter()
    m = GeoMixture([ContextMix(4), GRUByteModel(**gru_kw)], **kw)
    bits = ideal_bits(data, m)
    print(f"geomix {s:28s} {bits/len(data):.3f} bpb   final W {np.round(m.W,2).tolist()}  {time.perf_counter()-t:.0f}s", flush=True)
