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


@pytest.mark.parametrize("seed", SEEDS)
def test_kalman_update(seed):
    np.random.seed(seed)
    state, cov = _random_state_and_covariance()
    loadings, measurement, meas_sd = _random_loadings_measurement_and_meas_sd(state)
    dim = len(state)

    fp_filter = KalmanFilter(dim_x=dim, dim_z=1)
    fp_filter.x = state.reshape(dim, 1)
    fp_filter.F = np.eye(dim)
    fp_filter.H = loadings.reshape(1, dim)
    fp_filter.P = cov
    fp_filter.R = meas_sd ** 2

    fp_filter.update(measurement)

    expected_cov = fp_filter.P
    expected_state = fp_filter.x.flatten()
    expected_loglike = fp_filter.log_likelihood

    sm_states, sm_chols = _filterpy_to_skillmodels(state, cov)

    calc_states, calc_chols, calc_weights, calc_loglikes, _ = kalman_update(
        states=sm_states,
        upper_chols=sm_chols,
        loadings=jnp.array(loadings),
        control_params=jnp.ones(2),
        meas_sd=meas_sd,
        # plus 1 for the effect of the control variables
        measurements=jnp.array([measurement]) + 1,
        controls=jnp.ones((1, 2)) * 0.5,
        log_mixture_weights=jnp.ones((1, 1)),
        not_missing=jnp.array([True]),
        debug=False,
    )
    calculated_state = calc_states.flatten()
    calculated_cov = calc_chols.reshape(dim, dim).T @ calc_chols.reshape(dim, dim)
    calculated_loglike = calc_loglikes[0]

    aaae(calculated_state, expected_state)
    aaae(calculated_loglike, expected_loglike)
    aaae(calculated_cov, expected_cov)


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

    calc_states, calc_chols, calc_weights, calc_loglikes, _ = kalman_update(
        states=states,
        upper_chols=chols,
        loadings=jnp.ones(n_states) * 2,
        control_params=jnp.ones(2),
        meas_sd=1,
        measurements=measurements,
        controls=jnp.ones((n_obs, 2)) * 0.5,
        log_mixture_weights=jnp.log(jnp.ones((n_obs, 2)) * 0.5),
        not_missing=jnp.array([True, False, False]),
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
    expected = JulierSigmaPoints(n=len(state), kappa=2).sigma_points(state, cov)
    sm_state, sm_chol = _filterpy_to_skillmodels(state, cov)
    scaling_factor = np.sqrt(len(state) + 2)
    calculated = _calculate_sigma_points(sm_state, sm_chol, scaling_factor)
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

    def scale_and_sum(sigma_points, params):
        return (sigma_points * params[0]).sum(axis=-1)

    def constant(sigma_points, params):
        raise NotImplementedError

    transition_functions = (("scale_and_sum", scale_and_sum), ("constant", constant))

    trans_coeffs = (jnp.array([2]), jnp.array([]))

    anch_scaling = jnp.array([[1, 1], [2, 1]])

    anch_constants = np.array([[0, 0], [0, 0]])

    expected = jnp.array([[[[3, 2], [7, 4], [11, 6], [15, 8], [19, 10]]]])

    calculated = _transform_sigma_points(
        sp, transition_functions, trans_coeffs, anch_scaling, anch_constants
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

    sm_state, sm_chol = _filterpy_to_skillmodels(state, cov)
    scaling_factor, weights = calculate_sigma_scaling_factor_and_weights(dim, 2)
    transition_functions = (("linear", linear) for i in range(dim))
    trans_coeffs = (jnp.array(trans_mat[i]) for i in range(dim))
    anch_scaling = jnp.ones((2, dim))
    anch_constants = jnp.zeros((2, dim))

    calc_states, calc_chols = kalman_predict(
        sm_state,
        sm_chol,
        scaling_factor,
        weights,
        transition_functions,
        trans_coeffs,
        jnp.array(shock_sds),
        anch_scaling,
        anch_constants,
    )

    aaae(calc_states.flatten(), expected_state.flatten())
    aaae(calc_chols[0, 0].T @ calc_chols[0, 0], expected_cov)


# ======================================================================================
# Helper function to generate inputs and convert them between filterpy and skillmodels
# ======================================================================================


def _random_state_and_covariance():
    dim = np.random.randint(low=1, high=10)
    factorized = np.random.uniform(low=-1, high=3, size=(dim, dim))
    cov = factorized @ factorized.T * 0.5 + np.eye(dim)
    state = np.random.uniform(low=-5, high=5, size=dim)
    return state, cov


def _random_loadings_measurement_and_meas_sd(state):
    dim = len(state)
    loadings = np.random.uniform(size=dim)
    meas_sd = np.random.uniform()
    epsilon = np.random.normal(loc=0, scale=meas_sd)
    measurement = state @ loadings + epsilon
    return loadings, measurement, meas_sd


def _filterpy_to_skillmodels(state, cov):
    n_fac = len(state)
    sm_state = jnp.array(state).reshape(1, 1, n_fac)
    sm_chol = jnp.array(scipy.linalg.cholesky(cov)).reshape(1, 1, n_fac, n_fac)
    return sm_state, sm_chol
