import sys, time
import numpy as np
from neuralzip.codec import ideal_bits
from neuralzip.neural import GRUByteModel
from neuralzip.models import ContextMix, Order0
from neuralzip.mixing import GeoMixture
data = open(sys.argv[1],"rb").read()[:int(sys.argv[2])]
def mk(spec):
    ex = []
    for s in spec.split("+"):
        if s.startswith("ctx"): ex.append(ContextMix(int(s[3:])))
        elif s == "gru": ex.append(GRUByteModel())
        elif s == "order0": ex.append(Order0())
    return ex
for spec in sys.argv[3:]:
    lr = 0.005
    if ":" in spec: spec, lr = spec.split(":"); lr = float(lr)
    t=time.perf_counter(); m = GeoMixture(mk(spec), lr=lr)
    bits = ideal_bits(data, m)
    print(f"{spec:24s} lr={lr}  {bits/len(data):.3f} bpb  W={np.round(m.W[0],2).tolist()}  {time.perf_counter()-t:.0f}s", flush=True)
