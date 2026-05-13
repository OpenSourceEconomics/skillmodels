"""Estimator-agnostic transition-function helpers.

The core operation in any latent-factor model is

    state_next = (transition_function · anchor) (state_current)

with the per-period anchoring rescaling the latent factor onto the unit
of its anchor measurement before the transition runs and back to the
factor's own unit after. The CHS UKF, the AF Halton integrator, and the
AMN simulate-and-regress stage all need this composition; only the
CHS UKF additionally reshapes the inputs into sigma points.

`apply_anchored_transition` is the shared core. CHS's
`transform_sigma_points` wraps it with a `(n_obs, n_mixtures, n_sigma,
n_fac)`-aware reshape. Code that just needs to push a flat
`(N, n_fac)` panel of states through one period's transition (e.g.
`simulate_dataset`) can call this helper directly.
"""

from collections.abc import Callable

from jax import Array


def apply_anchored_transition(
    states: Array,
    transition_func: Callable[[dict[str, Array], Array], Array],
    trans_coeffs: dict[str, Array],
    anchoring_scaling_factors: Array,
    anchoring_constants: Array,
) -> Array:
    """Anchor states, apply the transition, then unanchor the result.

    Args:
        states: Shape `(N, n_fac)`. Each row is one (obs x mixture x
            sigma-point) packed into a flat sample.
        transition_func: Vectorised transition `(trans_coeffs, anchored)
            -> anchored_next`. Must broadcast over the leading axis.
        trans_coeffs: Per-factor transition parameters dict for the
            current period.
        anchoring_scaling_factors: Shape `(2, n_fac)`. Row 0 is the
            input period's scaling, row 1 is the output period's.
        anchoring_constants: Shape `(2, n_fac)`. Same layout.

    Return:
        Shape `(N, n_observed)` where `n_observed` is the leading
        latent-factor block of `n_fac` (observed factors at the tail
        of the input are passed through to the transition but the
        unanchoring slice trims them off — same convention as the
        former `transform_sigma_points`).
    """
    anchored = states * anchoring_scaling_factors[0] + anchoring_constants[0]
    transformed_anchored = transition_func(trans_coeffs, anchored)

    n_observed = transformed_anchored.shape[-1]
    return (
        transformed_anchored - anchoring_constants[1][:n_observed]
    ) / anchoring_scaling_factors[1][:n_observed]
