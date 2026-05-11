"""Simplest augmented model with endogenous factors.

A minimal model with two latent factors (fac1, fac2) and one observed factor (of).
Factor fac2 is endogenous. Both factors use linear transition functions with two
periods. Used for testing endogenous factor augmentation.
"""

from skillmodels.common.model_spec import (
    EstimationOptions,
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
    estimation_options=EstimationOptions(
        bounds_distance=1e-8,
    ),
)
