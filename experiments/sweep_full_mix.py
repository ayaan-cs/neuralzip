import sys, time
import numpy as np
from neuralzip.codec import ideal_bits
from neuralzip.neural import GRUByteModel
from neuralzip.models import ContextMix
from neuralzip.mixing import GeoMixture
data = open(sys.argv[1],"rb").read()
LADDER = [0,1,2,3,4,5,6,8,12,16,24]
GRU = dict(beta1=0, beta2=0.9999, eps=1e-5, lr=4e-3, lr_decay=1500)
def build(spec):
    ex = []
    for s in spec.split("+"):
        if s == "ladder": ex.append(ContextMix(max_order=24, orders=LADDER, alpha=6))
        elif s == "ctx8": ex.append(ContextMix(max_order=8, alpha=2))
        elif s == "ctx8a6": ex.append(ContextMix(max_order=8, alpha=6))
        elif s == "gru": ex.append(GRUByteModel(**GRU))
        elif s == "gru_old": ex.append(GRUByteModel())
        elif s == "gru_rt": ex.append(GRUByteModel(retrain_every=50000, retrain_window=50000, retrain_lr_scale=0.5, **GRU))
    return ex
for spec in sys.argv[2:]:
    lr = 0.005
    if ":" in spec: spec, lr = spec.split(":"); lr = float(lr)
    t = time.perf_counter(); m = GeoMixture(build(spec), lr=lr)
    print(f"{spec:22s} mixlr={lr}  {ideal_bits(data, m)/len(data):.4f} bpb  W={np.round(m.W[0],2).tolist()}  {time.perf_counter()-t:.0f}s", flush=True)
