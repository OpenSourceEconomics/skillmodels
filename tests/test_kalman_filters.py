"""Tests for Kalman filters."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy
from filterpy.kalman import JulierSigmaPoints, KalmanFilter
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.kalman_filters import (
    _calculate_sigma_points,
    calculate_sigma_scaling_factor_and_weights,
    kalman_predict,
    kalman_update,
    linear_kalman_predict,
    transform_sigma_points,
)
from skillmodels.kalman_filters_debug import kalman_update as kalman_update_debug

jax.config.update("jax_enable_x64", True)

SEEDS = range(20)
UPDATE_FUNCS = [kalman_update, kalman_update_debug]


@pytest.mark.parametrize(("seed", "update_func"), product(SEEDS, UPDATE_FUNCS))
def test_kalman_update(seed, update_func) -> None:
    rng = np.random.default_rng(seed)
    dim = int(rng.integers(low=1, high=10))
    n_obs = 5
    n_mix = 2

    states = np.zeros((n_obs, n_mix, dim))
    covs = np.zeros((n_obs, n_mix, dim, dim))
    for i in range(n_obs):
        for j in range(n_mix):
            states[i, j], covs[i, j] = _random_state_and_covariance(rng, dim=dim)

    loadings, measurements, meas_sd = _random_loadings_measurements_and_meas_sd(
        rng, states
    )

    expected_states = np.zeros_like(states)
    expected_covs = np.zeros_like(covs)

    for i in range(n_obs):
        for j in range(n_mix):
            fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
            fp_filter.x = states[i, j].reshape(dim, 1)
            fp_filter.F = np.eye(dim)
            fp_filter.H = loadings.reshape(1, dim)
            fp_filter.P = covs[i, j]
            fp_filter.R = meas_sd**2

            fp_filter.update(measurements[i])

            expected_states[i, j] = fp_filter.x.flatten()
            expected_covs[i, j] = fp_filter.P

    sm_states, sm_chols = _convert_update_inputs_from_filterpy_to_skillmodels(
        states,
        covs,
    )
    results = update_func(
        states=sm_states,
        upper_chols=sm_chols,
        loadings=jnp.array(loadings),
        control_params=jnp.ones(2),
        meas_sd=meas_sd,
        # plus 1 for the effect of the control variables
        measurements=jnp.array(measurements) + 1,
        controls=jnp.ones((n_obs, 2)) * 0.5,
        log_mixture_weights=jnp.full((n_obs, n_mix), jnp.log(0.5)),
    )
    calculated_covs = np.matmul(np.transpose(results[1], axes=(0, 1, 3, 2)), results[1])

    aaae(results[0], expected_states)
    aaae(calculated_covs, expected_covs)


@pytest.mark.parametrize("update_func", UPDATE_FUNCS)
def test_kalman_update_with_missing(update_func) -> None:
    """State, cov and weights should not change, log likelihood should be zero."""
    n_mixtures = 2
    n_obs = 3
    n_states = 4
    states = jnp.arange(24).reshape(n_obs, n_mixtures, n_states)

    chols = jnp.array(
        np.full((n_obs, n_mixtures, n_states, n_states), np.eye(n_states)),
    )

    measurements = jnp.array([13, jnp.nan, jnp.nan])
    weights = jnp.log(jnp.ones((n_obs, n_mixtures)) * 0.5)

    controls = np.ones((n_obs, 2)) * 0.5
    controls[1:] = np.nan
    controls = jnp.array(controls)

    results = update_func(
        states=states,
        upper_chols=chols,
        loadings=jnp.ones(n_states) * 2,
        control_params=jnp.ones(2),
        meas_sd=1,
        measurements=measurements,
        controls=controls,
        log_mixture_weights=jnp.log(jnp.ones((n_obs, 2)) * 0.5),
    )
    # debug version has an extra return, so go through this hoop.
    calc_states, calc_chols, calc_weights, calc_loglikes = results[:4]

    aaae(calc_states[1:], states[1:])
    aaae(calc_chols[1:], chols[1:])
    aaae(calc_loglikes[1:], jnp.zeros(2))
    aaae(calc_weights[1:], weights[1:])
    assert (calc_weights[0] != weights[0]).all()
    assert calc_states.shape == states.shape
    assert calc_chols.shape == chols.shape
    assert calc_weights.shape == weights.shape


@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_points(seed: int) -> None:
    rng = np.random.default_rng(seed)
    state, cov = _random_state_and_covariance(rng)
    observed_factors = jnp.arange(2).reshape(1, 2)
    expected = JulierSigmaPoints(n=len(state), kappa=2).sigma_points(state, cov)
    observed_part = np.tile(observed_factors, len(expected)).reshape(-1, 2)
    expected = np.hstack([expected, observed_part])
    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor = np.sqrt(len(state) + 2)
    calculated = _calculate_sigma_points(
        sm_state,
        sm_chol,
        scaling_factor,
        observed_factors,
    )
    aaae(calculated.reshape(expected.shape), expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_scaling_factor_and_weights(seed) -> None:
    rng = np.random.default_rng(seed)
    dim = int(rng.integers(low=1, high=15))
    kappa = float(rng.uniform(low=0.5, high=5))
    # Test my assumption that weights for mean and cov are equal in the Julier algorithm
    expected_weights = JulierSigmaPoints(n=dim, kappa=kappa).Wm
    expected_weights2 = JulierSigmaPoints(n=dim, kappa=kappa).Wc
    aaae(expected_weights, expected_weights2)
    # Test my code
    calc_scaling, calc_weights = calculate_sigma_scaling_factor_and_weights(dim, kappa)
    aaae(calc_weights, expected_weights)
    assert calc_scaling == np.sqrt(dim + kappa)


def test_transformation_of_sigma_points() -> None:
    sp = jnp.arange(10).reshape(1, 1, 5, 2) + 1

    def f(params, states):
        return jnp.column_stack(
            [(states * params["fac1"][0]).sum(axis=1), states[..., 1]],
        )

    trans_coeffs = {"fac1": jnp.array([2]), "fac2": jnp.array([])}

    anch_scaling = jnp.array([[1, 1], [2, 1]])

    anch_constants = jnp.array([[0, 0], [0, 0]])

    expected = jnp.array([[[[3, 2], [7, 4], [11, 6], [15, 8], [19, 10]]]])

    calculated = transform_sigma_points(
        sigma_points=sp,
        transition_func=f,
        trans_coeffs=trans_coeffs,
        anchoring_scaling_factors=anch_scaling,
        anchoring_constants=anch_constants,
    )

    aaae(calculated, expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_against_linear_filterpy(seed) -> None:
    rng = np.random.default_rng(seed)
    state, cov = _random_state_and_covariance(rng)
    dim = len(state)
    trans_mat = rng.uniform(low=-1, high=1, size=(dim, dim))

    shock_sds = 0.5 * np.arange(dim) / dim

    fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
    fp_filter.x = state.reshape(dim, 1)
    fp_filter.F = trans_mat
    fp_filter.P = cov
    fp_filter.Q = np.diag(shock_sds**2)

    fp_filter.predict()
    expected_state = fp_filter.x
    expected_cov = fp_filter.P

    def linear(params, states):
        return jnp.dot(states, params)

    def transition_function(params, states):
        return jnp.column_stack([linear(params[f"fac{i}"], states) for i in range(dim)])

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)
    trans_coeffs = {f"fac{i}": jnp.array(trans_mat[i]) for i in range(dim)}
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))
    observed_factors = jnp.zeros((1, 0))

    calc_states, calc_chols = kalman_predict(
        transition_function,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        jnp.asarray(observed_factors),
    )

    aaae(calc_states.flatten(), expected_state.flatten())
    aaae(calc_chols[0, 0].T @ calc_chols[0, 0], expected_cov)


@pytest.mark.parametrize("seed", SEEDS)
def test_linear_kalman_predict_against_filterpy(seed) -> None:
    """Test linear_kalman_predict gives same result as filterpy's linear predict."""
    rng = np.random.default_rng(seed)
    state, cov = _random_state_and_covariance(rng)
    dim = len(state)
    trans_mat = rng.uniform(low=-1, high=1, size=(dim, dim + 1))
    # last column is the constant
    f_mat = trans_mat[:, :-1]
    c_vec = trans_mat[:, -1]

    shock_sds = 0.5 * np.arange(dim) / max(dim, 1)

    fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
    fp_filter.x = state.reshape(dim, 1)
    fp_filter.F = f_mat
    fp_filter.B = np.eye(dim)
    fp_filter.P = cov
    fp_filter.Q = np.diag(shock_sds**2)

    fp_filter.predict(u=c_vec.reshape(dim, 1))
    expected_state = fp_filter.x
    expected_cov = fp_filter.P

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)

    latent_factors = tuple(f"fac{i}" for i in range(dim))
    trans_coeffs = {
        f"fac{i}": jnp.array(np.append(trans_mat[i, :-1], trans_mat[i, -1]))
        for i in range(dim)
    }
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))
    observed_factors = jnp.zeros((1, 0))

    calc_states, calc_chols = linear_kalman_predict(
        None,  # transition_func (ignored)
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
        latent_factors=latent_factors,
        constant_factor_indices=frozenset(),
        n_all_factors=dim,
    )

    aaae(calc_states.flatten(), expected_state.flatten())
    aaae(calc_chols[0, 0].T @ calc_chols[0, 0], expected_cov)


@pytest.mark.parametrize("seed", SEEDS)
def test_linear_predict_matches_unscented_for_linear_model(seed) -> None:
    """Linear predict should give identical results to unscented for linear models."""
    rng = np.random.default_rng(seed)
    state, cov = _random_state_and_covariance(rng)
    dim = len(state)
    trans_mat = rng.uniform(low=-1, high=1, size=(dim, dim))

    shock_sds = 0.5 * np.arange(dim) / max(dim, 1)

    def linear_func(params, states):
        return jnp.dot(states, params)

    def transition_function(params, states):
        return jnp.column_stack(
            [linear_func(params[f"fac{i}"], states) for i in range(dim)]
        )

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)
    # For unscented: trans_coeffs values are just the row of the transition matrix
    trans_coeffs_unscented = {f"fac{i}": jnp.array(trans_mat[i]) for i in range(dim)}
    # For linear: trans_coeffs values have constant appended (0 for pure linear)
    trans_coeffs_linear = {
        f"fac{i}": jnp.array(np.append(trans_mat[i], 0.0)) for i in range(dim)
    }
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))
    observed_factors = jnp.zeros((1, 0))
    latent_factors = tuple(f"fac{i}" for i in range(dim))

    unscented_states, unscented_chols = kalman_predict(
        transition_function,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs_unscented,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
    )

    linear_states, linear_chols = linear_kalman_predict(
        None,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs_linear,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
        latent_factors=latent_factors,
        constant_factor_indices=frozenset(),
        n_all_factors=dim,
    )

    aaae(linear_states, unscented_states, decimal=5)
    aaae(
        linear_chols[0, 0].T @ linear_chols[0, 0],
        unscented_chols[0, 0].T @ unscented_chols[0, 0],
        decimal=5,
    )


def test_linear_predict_with_constant_factors() -> None:
    """Test that constant factors produce identity rows in F."""
    rng = np.random.default_rng(42)
    dim = 3
    state, cov = _random_state_and_covariance(rng, dim=dim)
    shock_sds = np.array([0.1, 0.0, 0.2])

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)

    # fac0: linear, fac1: constant, fac2: linear
    trans_coeffs = {
        "fac0": jnp.array([0.5, 0.3, 0.1, 0.2]),  # 3 coeffs + constant
        "fac1": jnp.array([]),  # constant factor has no params
        "fac2": jnp.array([0.1, 0.2, 0.8, -0.1]),
    }
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))
    observed_factors = jnp.zeros((1, 0))
    latent_factors = ("fac0", "fac1", "fac2")

    calc_states, _calc_chols = linear_kalman_predict(
        None,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
        latent_factors=latent_factors,
        constant_factor_indices=frozenset({1}),
        n_all_factors=dim,
    )

    # fac1 (constant) should remain unchanged
    aaae(calc_states[0, 0, 1], state[1])

    # fac0 should be linear combination + constant
    expected_fac0 = 0.5 * state[0] + 0.3 * state[1] + 0.1 * state[2] + 0.2
    aaae(calc_states[0, 0, 0], expected_fac0)


def test_linear_predict_with_observed_factors() -> None:
    """Test that observed factors are used correctly as extra columns in F."""
    rng = np.random.default_rng(42)
    n_latent = 2
    n_observed = 1
    state, cov = _random_state_and_covariance(rng, dim=n_latent)
    shock_sds = np.array([0.1, 0.2])

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(n_latent, 2)

    observed_val = 3.0
    observed_factors = jnp.array([[observed_val]])

    # fac0 depends on both latent + observed, fac1 depends only on latent
    trans_coeffs = {
        "fac0": jnp.array([0.5, 0.3, 0.2, 0.1]),  # 2 latent + 1 observed + constant
        "fac1": jnp.array([0.1, 0.9, 0.0, 0.0]),
    }
    anch_scaling = jnp.ones((2, n_latent + n_observed))
    anch_constants = jnp.zeros((2, n_latent + n_observed))
    latent_factors = ("fac0", "fac1")

    calc_states, _calc_chols = linear_kalman_predict(
        None,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
        latent_factors=latent_factors,
        constant_factor_indices=frozenset(),
        n_all_factors=n_latent + n_observed,
    )

    expected_fac0 = 0.5 * state[0] + 0.3 * state[1] + 0.2 * observed_val + 0.1
    expected_fac1 = 0.1 * state[0] + 0.9 * state[1] + 0.0 * observed_val + 0.0
    aaae(calc_states[0, 0, 0], expected_fac0)
    aaae(calc_states[0, 0, 1], expected_fac1)


def test_linear_predict_with_wide_anchoring_arrays() -> None:
    """Regression: anchoring arrays have n_all columns, not just n_latent.

    At runtime, `parse_params` produces anchoring arrays of shape
    `(n_aug_periods, n_all_factors)` — latent columns followed by observed-factor
    columns (scaling=1, constant=0). This test uses that shape to verify
    `linear_kalman_predict` slices correctly.
    """
    rng = np.random.default_rng(42)
    n_latent = 2
    n_observed = 1
    n_all = n_latent + n_observed
    state, cov = _random_state_and_covariance(rng, dim=n_latent)
    shock_sds = np.array([0.1, 0.2])

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(n_latent, 2)

    observed_val = 3.0
    observed_factors = jnp.array([[observed_val]])

    trans_coeffs = {
        "fac0": jnp.array([0.5, 0.3, 0.2, 0.1]),
        "fac1": jnp.array([0.1, 0.9, 0.0, 0.0]),
    }
    # Shape (2, n_all) — matches what parse_params returns at runtime
    anch_scaling = jnp.ones((2, n_all))
    anch_constants = jnp.zeros((2, n_all))
    latent_factors = ("fac0", "fac1")

    calc_states, _calc_chols = linear_kalman_predict(
        None,
        sm_state,
        sm_chol,
        float(scaling_factor),
        weights,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
        latent_factors=latent_factors,
        constant_factor_indices=frozenset(),
        n_all_factors=n_all,
    )

    expected_fac0 = 0.5 * state[0] + 0.3 * state[1] + 0.2 * observed_val + 0.1
    expected_fac1 = 0.1 * state[0] + 0.9 * state[1] + 0.0 * observed_val + 0.0
    aaae(calc_states[0, 0, 0], expected_fac0)
    aaae(calc_states[0, 0, 1], expected_fac1)


def _random_state_and_covariance(rng, dim=None):
    if dim is None:
        dim = rng.integers(low=1, high=10)
    factorized = rng.uniform(low=-1, high=3, size=(dim, dim))
    cov = factorized @ factorized.T * 0.5 + np.eye(dim)
    state = rng.uniform(low=-5, high=5, size=dim)
    return state, cov


def _random_loadings_measurements_and_meas_sd(rng, state):
    n_obs, _n_mix, dim = state.shape
    loadings = rng.uniform(size=dim)
    meas_sd = rng.uniform()
    epsilon = rng.normal(loc=0, scale=meas_sd, size=(n_obs))
    measurement = (state @ loadings).sum(axis=1) + epsilon
    return loadings, measurement, meas_sd


def _convert_update_inputs_from_filterpy_to_skillmodels(state, cov):
    n_obs, n_mix, _n_fac = state.shape
    sm_state = jnp.array(state)
    sm_chol = np.zeros_like(cov)
    for i in range(n_obs):
        for j in range(n_mix):
            sm_chol[i, j] = scipy.linalg.cholesky(cov[i, j])
    sm_chol = jnp.array(sm_chol)
    return sm_state, sm_chol


def _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov):
    n_fac = len(state)
    sm_state = jnp.array(state).reshape(1, 1, n_fac)
    sm_chol = jnp.array(scipy.linalg.cholesky(cov)).reshape(1, 1, n_fac, n_fac)
    return sm_state, sm_chol


def test_sigma_points_multiple_mixtures() -> None:
    """Sigma points should work with n_mixtures >= 2."""
    n_obs = 2
    n_mixtures = 2
    n_states = 3
    n_observed = 2
    n_sigma = 2 * n_states + 1

    rng = np.random.default_rng(42)
    states = jnp.array(rng.standard_normal((n_obs, n_mixtures, n_states)))
    upper_chols = jnp.array(
        np.tile(np.eye(n_states), (n_obs, n_mixtures, 1, 1)),
    )
    observed_factors = jnp.array(rng.standard_normal((n_obs, n_observed)))
    scaling_factor = float(jnp.sqrt(n_states + 2))

    result = _calculate_sigma_points(
        states=states,
        upper_chols=upper_chols,
        scaling_factor=scaling_factor,
        observed_factors=observed_factors,
    )

    # Check output shape
    assert result.shape == (n_obs, n_mixtures, n_sigma, n_states + n_observed)

    # Observed columns should be constant across the sigma dimension
    for obs in range(n_obs):
        for mix in range(n_mixtures):
            observed_slice = result[obs, mix, :, n_states:]
            expected = jnp.broadcast_to(
                observed_factors[obs],
                (n_sigma, n_observed),
            )
            aaae(observed_slice, expected)
