"""Tests for the log_ces_general NLS seed in AMN Stage 3 (Pro F4).

`log_ces_general` names its exponents `sigma_<factor>` and its outside
coefficient `tfp`. The generic seed previously recognized only the exact names
`phi`/`rho`/`sigma`, so every parameter started at 0 and the transition
evaluated `tfp * log(sum gamma_i ...) = 0 * log(0) = NaN`, which made
scipy.least_squares raise "Residuals are not finite in the initial point". The
seed must recognize `sigma_*` and `tfp` and start the gammas strictly positive.
"""

import math

import jax.numpy as jnp
import numpy as np

from skillmodels.amn.simulate_and_regress import (
    _fit_generic_nls,
    _seed_generic_nls_theta0,
)
from skillmodels.common.transition_functions import log_ces_general


def test_log_ces_general_seed_is_finite() -> None:
    names = ("skills", "investment", "sigma_skills", "sigma_investment", "tfp")
    theta0 = _seed_generic_nls_theta0(names, {}, n_unknowns=len(names))
    # gammas strictly positive (so log(gamma) is finite).
    assert theta0[0] > 0.0
    assert theta0[1] > 0.0
    # exponents and outside coefficient nonzero.
    assert theta0[2] != 0.0
    assert theta0[3] != 0.0
    assert theta0[4] != 0.0
    # the transition evaluates finite at the seed.
    val = float(log_ces_general(jnp.asarray([1.0, 2.0]), jnp.asarray(theta0)))
    assert math.isfinite(val)


def test_fit_generic_nls_log_ces_general_does_not_blow_up() -> None:
    # Synthetic data from a known transformed CES; the fitter must start finite
    # and return a finite residual_sd (Pro F4 reproduction).
    g1, g2, s1, s2, tfp = 0.65, 0.35, -0.25, -0.5, -2.0
    rng = np.random.default_rng(0)
    states = rng.normal(size=(400, 2))
    y = tfp * np.log(g1 * np.exp(states[:, 0] * s1) + g2 * np.exp(states[:, 1] * s2))
    names = ("skills", "investment", "sigma_skills", "sigma_investment", "tfp")
    params, resid_sd = _fit_generic_nls(log_ces_general, names, y, states)
    assert math.isfinite(resid_sd)
    assert all(math.isfinite(v) for v in params.values())
