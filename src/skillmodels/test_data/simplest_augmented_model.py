"""Simplest augmented model with endogenous factors.

A minimal model with two latent factors (fac1, fac2) and one observed factor (of).
Factor fac2 is endogenous. Both factors use linear transition functions with two
periods. Used for testing endogenous factor augmentation.
"""

from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)

SIMPLEST_AUGMENTED_MODEL = ModelSpec(
    factors={
        "fac1": FactorSpec(
            measurements=(("var",), ("var",)),
            normalizations=Normalizations(
                loadings=({"var": 1}, {"var": 1}),
                intercepts=({}, {}),
            ),
            transition_function="linear",
        ),
        "fac2": FactorSpec(
            measurements=(("inv",), ("inv",)),
            normalizations=Normalizations(
                loadings=({"inv": 1}, {"inv": 1}),
                intercepts=({}, {}),
            ),
            is_endogenous=True,
            transition_function="linear",
        ),
    },
    observed_factors=("of",),
)

# CHS options used alongside SIMPLEST_AUGMENTED_MODEL in tests. Tests using
# this fixture exercise CHS plumbing rather than full estimation; opt into
# the cheap Spearman start-value path so collection stays fast. End-user
# defaults remain "amn".
SIMPLEST_AUGMENTED_MODEL_CHS_OPTIONS = CHSEstimationOptions(
    bounds_distance=1e-8,
    start_params_strategy="spearman",
)
