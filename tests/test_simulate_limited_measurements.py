"""Tests for limited-dependent-variable measurement simulation.

`measurements_from_states` builds the linear predictor `eta = controls @ b +
states @ lambda'` and then draws each measurement according to its family:

- Gaussian: `eta + N(0, sigma^2)`.
- Probit: `1{eta + N(0,1) >= 0}`, i.e. Bernoulli(`Phi(eta)`).
- Tobit: `clip(eta + N(0, sigma^2), lower, upper)`.

Passing no family arrays keeps the original all-Gaussian path unchanged.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from skillmodels.common.measurement_models import MeasurementFamily
from skillmodels.common.simulate_data import measurements_from_states

_INF = math.inf


def _single_measure(loading: float, eta_const: float):
    """One measurement loading 1:1 on one state plus a constant control."""
    states = np.ones((1, 1))  # placeholder; overwritten per test
    del states
    loadings = np.array([[loading]])
    control_params = np.array([[eta_const]])
    return loadings, control_params


def test_gaussian_path_unchanged_without_families() -> None:
    rng = np.random.default_rng(0)
    n_obs = 2000
    states = rng.normal(size=(n_obs, 1))
    controls = np.ones((n_obs, 1))
    loadings, control_params = _single_measure(1.0, 0.5)
    out = measurements_from_states(
        rng, states, controls, loadings, control_params, np.array([0.3])
    )
    # eta = 0.5 + state; residual ~ N(0, 0.3): mean(out - eta) ~ 0.
    eta = 0.5 + states[:, 0]
    assert np.std(out[:, 0] - eta) == pytest.approx(0.3, abs=0.02)


def test_probit_values_are_binary_and_match_frequency() -> None:
    rng = np.random.default_rng(1)
    n_obs = 200_000
    states = np.full((n_obs, 1), 1.0)
    controls = np.ones((n_obs, 1))
    loadings, control_params = _single_measure(0.0, 0.7)  # eta = 0.7 for everyone
    out = measurements_from_states(
        rng,
        states,
        controls,
        loadings,
        control_params,
        np.array([1.0]),
        families=np.array([int(MeasurementFamily.PROBIT)]),
        lowers=np.array([-_INF]),
        uppers=np.array([_INF]),
    )
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert out.mean() == pytest.approx(norm.cdf(0.7), abs=0.005)


def test_tobit_left_censoring_mass_and_interior() -> None:
    rng = np.random.default_rng(2)
    n_obs = 200_000
    states = np.full((n_obs, 1), 0.0)
    controls = np.ones((n_obs, 1))
    sigma = 1.0
    loadings, control_params = _single_measure(0.0, 0.3)  # eta = 0.3
    out = measurements_from_states(
        rng,
        states,
        controls,
        loadings,
        control_params,
        np.array([sigma]),
        families=np.array([int(MeasurementFamily.TOBIT)]),
        lowers=np.array([0.0]),
        uppers=np.array([_INF]),
    )[:, 0]
    # Nothing below the bound; a point mass at exactly 0.
    assert out.min() >= 0.0
    censored_frac = float(np.mean(out == 0.0))
    assert censored_frac == pytest.approx(norm.cdf((0.0 - 0.3) / sigma), abs=0.005)
    # Interior values are continuous (the censored mass aside).
    assert float(np.mean(out > 0.0)) == pytest.approx(1.0 - censored_frac, abs=1e-9)


def test_simulation_is_reproducible_with_seed() -> None:
    loadings, control_params = _single_measure(1.0, 0.0)
    states = np.random.default_rng(7).normal(size=(100, 1))
    controls = np.ones((100, 1))
    sds = np.array([1.0])
    families = np.array([int(MeasurementFamily.PROBIT)])
    lowers = np.array([-_INF])
    uppers = np.array([_INF])

    def _draw(seed: int):
        return measurements_from_states(
            np.random.default_rng(seed),
            states,
            controls,
            loadings,
            control_params,
            sds,
            families=families,
            lowers=lowers,
            uppers=uppers,
        )

    np.testing.assert_array_equal(_draw(3), _draw(3))
