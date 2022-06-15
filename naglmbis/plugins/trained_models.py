from qubekit.nonbonded import LennardJones612
from qubekit.nonbonded.protocols import (
    br_base,
    c_base,
    cl_base,
    f_base,
    h_base,
    n_base,
    o_base,
    s_base,
)

# build a QUBEKit LJ class with pre-optimised Rfree parameters
model_v1 = LennardJones612(
    free_parameters={
        "H": h_base(r_free=1.765),
        "C": c_base(r_free=2.067),
        "N": n_base(r_free=1.688),
        "O": o_base(r_free=1.653),
        "X": h_base(r_free=1.211),
        "Cl": cl_base(r_free=1.935),
        "S": s_base(r_free=2.043),
        "F": f_base(r_free=1.642),
        "Br": br_base(r_free=2.037),
    },
    alpha=1.166,
    beta=0.479,
)

trained_models = {1: model_v1}
