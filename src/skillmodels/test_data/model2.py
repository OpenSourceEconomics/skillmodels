"""Model 2 from the replication files of Cunha, Heckman, and Schennach (2010).

This model has three latent factors (fac1, fac2, fac3) observed over 8 periods,
with CES, linear, and constant transition functions respectively. It includes
anchoring of fac1 to outcome Q1 and a single control variable x1.
"""

from skillmodels.model_spec import (
    AnchoringSpec,
    EstimationOptions,
    FactorSpec,
    ModelSpec,
    Normalizations,
)

MODEL2 = ModelSpec(
    factors={
        "fac1": FactorSpec(
            measurements=(("y1", "y2", "y3"),) * 8,
            normalizations=Normalizations(
                loadings=({"y1": 1},) * 8,
                intercepts=({},) * 8,
            ),
            transition_function="log_ces",
        ),
        "fac2": FactorSpec(
            measurements=(("y4", "y5", "y6"),) * 8,
            normalizations=Normalizations(
                loadings=({"y4": 1},) * 8,
                intercepts=({},) * 8,
            ),
            transition_function="linear",
        ),
        "fac3": FactorSpec(
            measurements=(("y7", "y8", "y9"),) + ((),) * 7,
            normalizations=Normalizations(
                loadings=({"y7": 1},) + ({},) * 7,
                intercepts=({},) * 8,
            ),
            transition_function="constant",
        ),
    },
    anchoring=AnchoringSpec(
        outcomes={"fac1": "Q1"},
        free_controls=True,
        free_constant=True,
        free_loadings=True,
        ignore_constant_when_anchoring=True,
    ),
    controls=("x1",),
    stagemap=(0, 0, 0, 0, 0, 0, 0),
    estimation_options=EstimationOptions(
        robust_bounds=True,
        bounds_distance=0.001,
        n_mixtures=1,
    ),
)
