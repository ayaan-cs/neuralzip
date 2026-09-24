import sys, time
import neuralzip  # noqa: F401  -- before numpy: pins BLAS to one thread
import numpy as np
from neuralzip.registry import MODELS
from neuralzip.neural import GRUByteModel
from neuralzip.models import ContextMix
data = open(sys.argv[1],"rb").read()
seg = int(sys.argv[2])
specs = {"ctx4": lambda: ContextMix(4)}
for s in sys.argv[3:]:
    specs[s] = (lambda s=s: GRUByteModel(**eval("dict("+s+")")))
for name, mk in specs.items():
    m = mk(); t=time.perf_counter(); out=[]; bits=0.0; total=0.0
    for i,b in enumerate(data):
        p=m.predict(); bits -= np.log2(max(p[b],1e-12)); m.update(b)
        if (i+1)%seg==0: out.append(bits/seg); total+=bits; bits=0
    total += bits
    print(f"{name:28s} overall {total/len(data):.3f}  " + " ".join(f"{x:.2f}" for x in out) + f"   {time.perf_counter()-t:.0f}s", flush=True)
