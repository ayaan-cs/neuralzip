"""Name -> factory for every model the CLI and benchmark can use."""
from .mixing import GeoMixture
from .models import ContextMix, Order0
from .neural import GRUByteModel

# Context orders modelled by the headline context model: every order up to 6,
# then sparse jumps.  Long contexts are almost always deterministic, so the
# high orders are cheap and catch repeated phrases (the PPM* observation).
LADDER = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24]

# Adam settings from NNCP (Bellard 2019): beta1 = 0 turns Adam into
# bias-corrected RMSProp, which tracks a non-stationary stream better than
# momentum does.
GRU_KW = dict(beta1=0.0, beta2=0.9999, eps=1e-5, lr=4e-3, lr_decay=1500)


def ladder():
    return ContextMix(max_order=LADDER[-1], orders=LADDER, alpha=6.0)


MODELS = {
    "order0": lambda: Order0(),
    "ctx2": lambda: ContextMix(max_order=2),
    "ctx4": lambda: ContextMix(max_order=4),
    "ctx8": lambda: ContextMix(max_order=8),
    "ctx": ladder,
    "gru": lambda: GRUByteModel(**GRU_KW),
    # the headline model: context ladder + online GRU, geometrically mixed
    "nz": lambda: GeoMixture([ladder(), GRUByteModel(**GRU_KW)], lr=0.005),
    # same, plus NNCP-v2-style periodic retraining of the GRU (slower, no gain at <1 MB)
    "nz-retrain": lambda: GeoMixture([ladder(), GRUByteModel(retrain_every=50_000, retrain_window=50_000, **GRU_KW)], lr=0.005),
}
