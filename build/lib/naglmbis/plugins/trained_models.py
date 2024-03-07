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

# A second model trained on Mixture parameters against tip4pfb water
model_v1_mixture = LennardJones612(
    free_parameters={
        "H": h_base(r_free=1.887),
        "C": c_base(r_free=2.058),
        "N": n_base(r_free=1.631),
        "O": o_base(r_free=1.659),
        "X": h_base(r_free=0.978),
        "Cl": cl_base(r_free=1.868),
        "S": s_base(r_free=1.841),
        "F": f_base(r_free=1.644),
        "Br": br_base(r_free=1.932),
    },
    alpha=1.216,
    beta=0.487,
)

# Trained on mixture properties with tip4p-fb and bccs using nagl charge and volume v1 no polar h
model_v1_mixture_no_polar_bcc = LennardJones612(
    free_parameters={
        "H": h_base(r_free=1.825),
        "C": c_base(r_free=2.059),
        "N": n_base(r_free=1.636),
        "O": o_base(r_free=1.707),
        "Cl": cl_base(r_free=1.877),
        "S": s_base(r_free=1.811),
        "F": f_base(r_free=1.637),
        "Br": br_base(r_free=1.917),
    },
    lj_on_polar_h=False,
    alpha=1.165,
    beta=0.476,
)

# A model optimised with Mixture properties against tip4p-fb using espaloma-charge-0.0.8
model_v1_espaloma_mixture = LennardJones612(
    free_parameters={
        "H": h_base(r_free=1.925),
        "C": c_base(r_free=2.013),
        "N": n_base(r_free=1.807),
        "O": o_base(r_free=1.537),
        "X": h_base(r_free=1.256),
        "Cl": cl_base(r_free=1.866),
        "S": s_base(r_free=1.812),
        "F": f_base(r_free=1.627),
        "Br": br_base(r_free=1.969),
    },
    alpha=1.2,
    beta=0.502,
)

model_v2_espaloma_mixture_no_polar_h = LennardJones612(
    free_parameters={
        "H": h_base(r_free=1.868),
        "C": c_base(r_free=2.022),
        "N": n_base(r_free=1.835),
        "O": o_base(r_free=1.603),
        "Cl": cl_base(r_free=1.846),
        "S": s_base(r_free=1.810),
        "F": f_base(r_free=1.566),
        "Br": br_base(r_free=1.930),
    },
    lj_on_polar_h=False,
    alpha=1.129,
    beta=0.555,
)

trained_models = {
    1: model_v1,
    2: model_v1_mixture,
    3: model_v1_mixture_no_polar_bcc,
    "espaloma-v1": model_v1_espaloma_mixture,
    "espaloma-v2": model_v2_espaloma_mixture_no_polar_h,
}
