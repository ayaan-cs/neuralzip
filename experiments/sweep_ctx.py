import sys, time
from neuralzip.codec import ideal_bits
from neuralzip.models import ContextMix
data = open(sys.argv[1],"rb").read()[:int(sys.argv[2])]
for s in sys.argv[3:]:
    kw = eval("dict("+s+")"); t=time.perf_counter()
    print(f"ctx {s:30s} {ideal_bits(data, ContextMix(**kw))/len(data):.4f} bpb  {time.perf_counter()-t:.0f}s", flush=True)
