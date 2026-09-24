import sys, time
import neuralzip  # noqa: F401  -- before numpy: pins BLAS to one thread
import numpy as np
from neuralzip.codec import ideal_bits
from neuralzip.neural import GRUByteModel
from neuralzip.models import ContextMix
from neuralzip.mixing import Mixture
data = open("corpora/english.txt","rb").read()[:int(sys.argv[1])]
gru_kw = eval("dict("+sys.argv[2]+")")
for s in sys.argv[3:]:
    kw = eval("dict("+s+")")
    t=time.perf_counter()
    m = Mixture([ContextMix(4), GRUByteModel(**gru_kw)], **kw)
    bits = ideal_bits(data, m)
    print(f"mix {kw}  {bits/len(data):.3f} bpb   final weights {np.round(m.weights(),3)}  {time.perf_counter()-t:.0f}s", flush=True)
