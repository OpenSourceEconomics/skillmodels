import skillmodels.model_functions.transition_functions as tf
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
import pytest


# ======================================================================================
# linear
# ======================================================================================


@pytest.fixture
def setup_linear():
    nemf, nind, nsigma, nfac = 2, 10, 7, 3
    sigma_points = np.ones((nemf, nind, nsigma, nfac))
    sigma_points[1] *= 2
    sigma_points[:, :, 0, :] = 3
    sigma_points = sigma_points.reshape(nemf * nind * nsigma, nfac)

    args = {
        "sigma_points": sigma_points,
        "coeffs": np.array([0.5, 1.0, 1.5]),
        "included_positions": np.array([0, 1, 2]),
    }
    return args


@pytest.fixture
def expected_linear():
    nemf, nind, nsigma = 2, 10, 7
    expected_result = np.ones((nemf, nind, nsigma)) * 3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result.flatten()
    return expected_result


def test_linear(setup_linear, expected_linear):
    aaae(tf.linear(**setup_linear), expected_linear)


# ======================================================================================
# linear with constant
# ======================================================================================
@pytest.fixture
def expected_linear_with_constant():
    nemf, nind, nsigma = 2, 10, 7
    coeffs = [0.5, 1.0, 1.5, 0.5]
    expected_result = np.ones((nemf, nind, nsigma)) * 3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result.flatten()

    return coeffs[-1] + expected_result


def test_linear_with_constant(setup_linear, expected_linear):
    setup_linear["coeffs"] = [0.5, 1.0, 1.5, 0.5]
    expected_linear_with_constant = expected_linear + 0.5
    aaae(tf.linear_with_constant(**setup_linear), expected_linear_with_constant)


# ======================================================================================
# ar1
# ======================================================================================

@pytest.fixture
def setup_ar1_transition_equation():
    # nemf_nind_nsigma is nemf*nind*nsigma
    nemf_nind_nsigma = 140

    args = {
        "sigma_points": np.ones((nemf_nind_nsigma, 3)),
        "coeffs": np.array([3]),
        "included_positions": [1],
    }
    return args


@pytest.fixture
def expected_ar1_transition_equation():
    nemf_nind_nsigma = 140
    expected_result = np.ones(nemf_nind_nsigma) * 3

    return expected_result


def test_ar1_transition_equation(
    setup_ar1_transition_equation, expected_ar1_transition_equation
):
    aaae(tf.ar1(**setup_ar1_transition_equation), expected_ar1_transition_equation)


# ======================================================================================
# log_ces
# ======================================================================================

@pytest.fixture
def setup_log_ces():
    nsigma = 5
    sigma_points = np.array([[3, 7.5]] * nsigma)

    args = {
        "sigma_points": sigma_points,
        "coeffs": np.array([0.4, 0.6, 2]),
        "included_positions": [0, 1],
    }
    return args


@pytest.fixture
def expected_log_ces():
    nsigma = 5
    expected_result = np.ones(nsigma) * 7.244628323025
    return expected_result


def test_log_ces(setup_log_ces, expected_log_ces):
    aaae(tf.log_ces(**setup_log_ces), expected_log_ces)


# ======================================================================================
# translog
# ======================================================================================

@pytest.fixture
def setup_translog():
    sigma_points = np.array(
        [
            [2, 0, 5, 0],
            [0, 3, 5, 0],
            [0, 0, 7, 4],
            [0, 0, 1, 0],
            [1, 1, 10, 1],
            [0, -3, -100, 0],
            [-1, -1, -1, -1],
            [1.5, -2, 30, 1.8],
            [12, -34, 50, 48],
        ]
    )

    coeffs = np.array([0.2, 0.1, 0.12, 0.08, 0.05, 0.04, 0.03, 0.06, 0.05, 0.04])

    included_positions = [0, 1, 3]

    args = {
        "sigma_points": sigma_points,
        "coeffs": coeffs,
        "included_positions": included_positions,
    }
    return args


@pytest.fixture
def expected_translog():
    expected_result = np.array([0.76, 0.61, 1.32, 0.04, 0.77, 0.01, -0.07, 0.56, 70.92])

    return expected_result


def test_translog(setup_translog, expected_translog):
    aaae(tf.translog(**setup_translog), expected_translog)
