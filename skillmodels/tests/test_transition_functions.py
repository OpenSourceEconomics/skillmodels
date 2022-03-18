import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.transition_functions import linear
from skillmodels.transition_functions import log_ces
from skillmodels.transition_functions import translog

config.update("jax_enable_x64", True)
# ======================================================================================
# linear
# ======================================================================================


@pytest.fixture
def setup_linear():
    nmixtures, nind, nsigma, nfac = 2, 10, 7, 3
    sigma_points = np.ones((nmixtures, nind, nsigma, nfac))
    sigma_points[1] *= 2
    sigma_points[:, :, 0, :] = 3

    args = {
        "states": jnp.array(sigma_points),
        "params": jnp.array([0.5, 1.0, 1.5, 0.5]),
    }
    return args


@pytest.fixture
def expected_linear():
    nmixtures, nind, nsigma = 2, 10, 7
    expected_result = np.ones((nmixtures, nind, nsigma)) * 3
    expected_result[1, :, :] *= 2
    expected_result[:, :, 0] = 9
    expected_result = expected_result
    expected_result += 0.5
    return expected_result


def test_linear(setup_linear, expected_linear):
    aaae(linear(**setup_linear), expected_linear)


# ======================================================================================
# translog
# ======================================================================================


@pytest.fixture
def setup_translog():
    sigma_points = np.array(
        [
            [
                [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
                [[0, 0, 0], [1, 1, 1], [0, -3, 0]],
                [[-1, -1, -1], [1.5, -2, 1.8], [12, -34, 48]],
            ]
        ]
    )

    params = np.array(
        [
            # linear terms
            0.2,
            0.1,
            0.12,
            # square terms
            0.08,
            0.04,
            0.05,
            # interactions: The order is 0-1, 0-2, 1-2
            0.05,
            0.03,
            0.06,
            # constant
            0.04,
        ]
    )

    args = {"states": sigma_points, "params": params}
    return args


@pytest.fixture
def expected_translog():
    expected_result = np.array(
        [[[0.76, 0.61, 1.32], [0.04, 0.77, 0.01], [-0.07, 0.56, 70.92]]]
    )

    expected_result = np.array(
        [
            [
                [0.7600, 0.7000, 1.3200],
                [0.0400, 0.7700, 0.1000],
                [-0.0700, 0.5730, 76.7200],
            ]
        ]
    )

    return expected_result


def test_translog(setup_translog, expected_translog):
    aaae(translog(**setup_translog), expected_translog)


# ======================================================================================
# log_ces
# ======================================================================================


@pytest.fixture
def setup_log_ces():
    nsigma = 5
    sigma_points = np.array([[3, 7.5]] * nsigma)

    args = {"states": jnp.array(sigma_points), "params": jnp.array([0.4, 0.6, 2])}
    return args


@pytest.fixture
def expected_log_ces():
    nsigma = 5
    expected_result = np.ones(nsigma) * 7.244628323025
    return expected_result


def test_log_ces(setup_log_ces, expected_log_ces):
    aaae(log_ces(**setup_log_ces), expected_log_ces)


def test_where_all_but_one_gammas_are_zero():
    """This has to be tested, becaus it leads to an underflow in the log step."""
    sigma_points = jnp.ones((2, 2, 3))
    params = jnp.array([0, 0, 1, -0.5])
    calculated = log_ces(sigma_points, params)
    expected = jnp.ones((2, 2))
    aaae(calculated, expected)
