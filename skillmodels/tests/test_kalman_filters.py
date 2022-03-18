from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
import scipy
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import KalmanFilter
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.kalman_filters import _calculate_sigma_points
from skillmodels.kalman_filters import _transform_sigma_points
from skillmodels.kalman_filters import calculate_sigma_scaling_factor_and_weights
from skillmodels.kalman_filters import kalman_predict
from skillmodels.kalman_filters import kalman_update

config.update("jax_enable_x64", True)

# ======================================================================================
# Test Kalman Update with random state and cov againts filterpy
# ======================================================================================

SEEDS = range(20)
UPDATE_FUNCS = [kalman_update]
TEST_CASES = product(SEEDS, UPDATE_FUNCS)


@pytest.mark.parametrize("seed, update_func", TEST_CASES)
def test_kalman_update(seed, update_func):
    np.random.seed(seed)
    dim = np.random.randint(low=1, high=10)
    n_obs = 5
    n_mix = 2

    states = np.zeros((n_obs, n_mix, dim))
    covs = np.zeros((n_obs, n_mix, dim, dim))
    for i in range(n_obs):
        for j in range(n_mix):
            states[i, j], covs[i, j] = _random_state_and_covariance(dim=dim)

    loadings, measurements, meas_sd = _random_loadings_measurements_and_meas_sd(states)

    expected_states = np.zeros_like(states)
    expected_covs = np.zeros_like(covs)

    for i in range(n_obs):
        for j in range(n_mix):
            fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
            fp_filter.x = states[i, j].reshape(dim, 1)
            fp_filter.F = np.eye(dim)
            fp_filter.H = loadings.reshape(1, dim)
            fp_filter.P = covs[i, j]
            fp_filter.R = meas_sd ** 2

            fp_filter.update(measurements[i])

            expected_states[i, j] = fp_filter.x.flatten()
            expected_covs[i, j] = fp_filter.P

    sm_states, sm_chols = _convert_update_inputs_from_filterpy_to_skillmodels(
        states, covs
    )

    calc_states, calc_chols, calc_weights, calc_loglikes, _ = update_func(
        states=sm_states,
        upper_chols=sm_chols,
        loadings=jnp.array(loadings),
        control_params=jnp.ones(2),
        meas_sd=meas_sd,
        # plus 1 for the effect of the control variables
        measurements=jnp.array(measurements) + 1,
        controls=jnp.ones((n_obs, 2)) * 0.5,
        log_mixture_weights=jnp.full((n_obs, n_mix), jnp.log(0.5)),
        debug=False,
    )
    calculated_covs = np.matmul(np.transpose(calc_chols, axes=(0, 1, 3, 2)), calc_chols)

    aaae(calc_states, expected_states)
    aaae(calculated_covs, expected_covs)


# ======================================================================================
# Test Kalman Update with missings
# ======================================================================================


def test_kalman_update_with_missing():
    """State, cov and weights should not change, log likelihood should be zero."""

    n_mixtures = 2
    n_obs = 3
    n_states = 4
    states = jnp.arange(24).reshape(n_obs, n_mixtures, n_states)

    chols = jnp.array(
        np.full((n_obs, n_mixtures, n_states, n_states), np.eye(n_states))
    )

    measurements = jnp.array([13, jnp.nan, jnp.nan])
    weights = jnp.log(jnp.ones((n_obs, n_mixtures)) * 0.5)

    controls = np.ones((n_obs, 2)) * 0.5
    controls[1:] = np.nan
    controls = jnp.array(controls)

    calc_states, calc_chols, calc_weights, calc_loglikes, _ = kalman_update(
        states=states,
        upper_chols=chols,
        loadings=jnp.ones(n_states) * 2,
        control_params=jnp.ones(2),
        meas_sd=1,
        measurements=measurements,
        controls=controls,
        log_mixture_weights=jnp.log(jnp.ones((n_obs, 2)) * 0.5),
        debug=False,
    )

    aaae(calc_states[1:], states[1:])
    aaae(calc_chols[1:], chols[1:])
    aaae(calc_loglikes[1:], jnp.zeros(2))
    aaae(calc_weights[1:], weights[1:])
    assert (calc_weights[0] != weights[0]).all()
    assert calc_states.shape == states.shape
    assert calc_chols.shape == chols.shape
    assert calc_weights.shape == weights.shape


# ======================================================================================
# test generation of sigma points
# ======================================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_points(seed):
    np.random.seed(seed)
    state, cov = _random_state_and_covariance()
    observed_factors = np.arange(2).reshape(1, 2)
    expected = JulierSigmaPoints(n=len(state), kappa=2).sigma_points(state, cov)
    observed_part = np.tile(observed_factors, len(expected)).reshape(-1, 2)
    expected = np.hstack([expected, observed_part])
    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor = np.sqrt(len(state) + 2)
    calculated = _calculate_sigma_points(
        sm_state, sm_chol, scaling_factor, observed_factors
    )
    aaae(calculated.reshape(expected.shape), expected)


# ======================================================================================
# Test sigma weights and scaling factor
# ======================================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_scaling_factor_and_weights(seed):
    np.random.seed
    dim = np.random.randint(low=1, high=15)
    kappa = np.random.uniform(low=0.5, high=5)
    # Test my assumption that weights for mean and cov are equal in the Julier algorithm
    expected_weights = JulierSigmaPoints(n=dim, kappa=kappa).Wm
    expected_weights2 = JulierSigmaPoints(n=dim, kappa=kappa).Wc
    aaae(expected_weights, expected_weights2)
    # Test my code
    calc_scaling, calc_weights = calculate_sigma_scaling_factor_and_weights(dim, kappa)
    aaae(calc_weights, expected_weights)
    assert calc_scaling == np.sqrt(dim + kappa)


# ======================================================================================
# test transformation of sigma points
# ======================================================================================


def test_transformation_of_sigma_points():
    sp = jnp.arange(10).reshape(1, 1, 5, 2) + 1

    def f(sigma_points, fac2, params):
        out = jnp.column_stack([(sigma_points * params["fac1"][0]).sum(axis=1), fac2])
        return out

    transition_info = {"func": f, "columns": {"fac2": 1}}

    trans_coeffs = {"fac1": jnp.array([2]), "fac2": jnp.array([])}

    anch_scaling = jnp.array([[1, 1], [2, 1]])

    anch_constants = np.array([[0, 0], [0, 0]])

    expected = jnp.array([[[[3, 2], [7, 4], [11, 6], [15, 8], [19, 10]]]])

    calculated = _transform_sigma_points(
        sp, transition_info, trans_coeffs, anch_scaling, anch_constants
    )

    aaae(calculated, expected)


# ======================================================================================
# test special case against linear predict from filterpy
# - anchoring scaling factors are 1
# - anchoring constants are 0
# - linear transition functions
# ======================================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_against_linear_filterpy(seed):
    np.random.seed(seed)
    state, cov = _random_state_and_covariance()
    dim = len(state)
    trans_mat = np.random.uniform(low=-1, high=1, size=(dim, dim))

    shock_sds = 0.5 * np.arange(dim) / dim

    fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
    fp_filter.x = state.reshape(dim, 1)
    fp_filter.F = trans_mat
    fp_filter.P = cov
    fp_filter.Q = np.diag(shock_sds ** 2)

    fp_filter.predict()
    expected_state = fp_filter.x
    expected_cov = fp_filter.P

    def linear(sigma_points, params):
        return np.dot(sigma_points, params)

    def transition_function(sigma_points, params):
        out = jnp.column_stack(
            [linear(sigma_points, params[f"fac{i}"]) for i in range(dim)]
        )
        return out

    sm_state, sm_chol = _convert_predict_inputs_from_filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)
    transition_info = {"func": transition_function, "columns": {}}
    trans_coeffs = {f"fac{i}": jnp.array(trans_mat[i]) for i in range(dim)}
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))
    observed_factors = jnp.zeros((1, 0))

    calc_states, calc_chols = kalman_predict(
        sm_state,
        sm_chol,
        scaling_factor,
        weights,
        transition_info,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
        observed_factors,
    )

    aaae(calc_states.flatten(), expected_state.flatten())
    aaae(calc_chols[0, 0].T @ calc_chols[0, 0], expected_cov)


# ======================================================================================
# Helper function to generate inputs and convert them between filterpy and skillmodels
# ======================================================================================


def _random_state_and_covariance(dim=None):
    if dim is None:
        dim = np.random.randint(low=1, high=10)
    factorized = np.random.uniform(low=-1, high=3, size=(dim, dim))
    cov = factorized @ factorized.T * 0.5 + np.eye(dim)
    state = np.random.uniform(low=-5, high=5, size=dim)
    return state, cov


def _random_loadings_measurements_and_meas_sd(state):
    n_obs, n_mix, dim = state.shape
    loadings = np.random.uniform(size=dim)
    meas_sd = np.random.uniform()
    epsilon = np.random.normal(loc=0, scale=meas_sd, size=(n_obs))
    measurement = (state @ loadings).sum(axis=1) + epsilon
    return loadings, measurement, meas_sd


def _convert_update_inputs_from_filterpy_to_skillmodels(state, cov):
    n_obs, n_mix, n_fac = state.shape
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
