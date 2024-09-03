import jax.numpy as jnp
import numpy as np
from jax import config
from numpy.testing import assert_array_almost_equal as aaae
from skillmodels.transition_functions import (
    constant,
    linear,
    log_ces,
    log_ces_general,
    params_log_ces_general,
    robust_translog,
    translog,
)

config.update("jax_enable_x64", True)


def test_linear():
    states = np.arange(3)
    params = np.array([0.1, 0.2, 0.3, 0.4])
    expected = 1.2
    aaae(linear(states, params), expected)


def test_translog():
    all_states = np.array(
        [
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 4],
            [0, 0, 0],
            [1, 1, 1],
            [0, -3, 0],
            [-1, -1, -1],
            [1.5, -2, 1.8],
            [12, -34, 48],
        ],
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
        ],
    )

    expected_translog = [0.76, 0.7, 1.32, 0.04, 0.77, 0.1, -0.07, 0.573, 76.72]

    for states, expected in zip(all_states, expected_translog, strict=False):
        calculated = translog(states, params)
        aaae(calculated, expected)


def test_log_ces():
    states = np.array([3, 7.5])
    params = jnp.array([0.4, 0.6, 2])
    expected = 7.244628323025
    calculated = log_ces(states, params)
    aaae(calculated, expected)


def test_where_all_but_one_gammas_are_zero():
    """This has to be tested, becaus it leads to an underflow in the log step."""
    states = jnp.ones(3)
    params = jnp.array([0, 0, 1, -0.5])
    calculated = log_ces(states, params)
    expected = 1.0
    aaae(calculated, expected)


def test_constant():
    assert constant("bla", "blubb") == "bla"


def test_robust_translog():
    all_states = np.array(
        [
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 4],
            [0, 0, 0],
            [1, 1, 1],
            [0, -3, 0],
            [-1, -1, -1],
            [1.5, -2, 1.8],
            [12, -34, 48],
        ],
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
        ],
    )

    expected_translog = [0.76, 0.7, 1.32, 0.04, 0.77, 0.1, -0.07, 0.573, 76.72]

    for states, expected in zip(all_states, expected_translog, strict=False):
        calculated = robust_translog(states, params)
        aaae(calculated, expected)


def test_log_ces_general():
    states = np.array([3, 7.5])
    params = jnp.array([0.4, 0.6, 2, 2, 0.5])
    expected = 7.244628323025
    calculated = log_ces_general(states, params)
    aaae(calculated, expected)


def test_log_ces_general_where_all_but_one_gammas_are_zero():
    """This has to be tested, becaus it leads to an underflow in the log step."""
    states = jnp.ones(3)
    params = jnp.array([0, 0, 1, -0.5, -0.5, -0.5, -2])
    calculated = log_ces_general(states, params)
    expected = 1.0
    aaae(calculated, expected)


def test_param_names_log_ces_general():
    factors = ["a", "b"]
    expected = ["a", "b", "sigma_a", "sigma_b", "tfp"]
    calculated = params_log_ces_general(factors)
    assert calculated == expected
