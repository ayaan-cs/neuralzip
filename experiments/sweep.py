import sys, time, itertools
import numpy as np
from neuralzip.codec import ideal_bits
from neuralzip.neural import GRUByteModel
data = open("corpora/english.txt","rb").read()[:int(sys.argv[1])]
for kw in [eval("dict(" + s + ")") for s in sys.argv[2:]]:
    t=time.perf_counter(); bits = ideal_bits(data, GRUByteModel(**kw)); dt=time.perf_counter()-t
    print(f"{kw}  {bits/len(data):.3f} bpb  {dt:.1f}s  {len(data)/dt/1000:.1f} KB/s", flush=True)
