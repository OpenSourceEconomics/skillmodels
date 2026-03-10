import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from skillmodels.transition_functions import (
    constant,
    constraints_log_ces,
    identity_constraints_linear,
    identity_constraints_linear_and_squares,
    identity_constraints_log_ces,
    identity_constraints_log_ces_general,
    identity_constraints_robust_translog,
    identity_constraints_translog,
    linear,
    linear_and_squares,
    log_ces,
    log_ces_general,
    params_constant,
    params_linear,
    params_linear_and_squares,
    params_log_ces,
    params_log_ces_general,
    params_robust_translog,
    params_translog,
    robust_translog,
    translog,
)

jax.config.update("jax_enable_x64", True)


def test_linear() -> None:
    states = jnp.arange(3)
    params = jnp.array([0.1, 0.2, 0.3, 0.4])
    expected = 1.2
    aaae(linear(states, params), expected)


def test_translog() -> None:
    all_states = jnp.array(
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

    params = jnp.array(
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
        calculated = translog(jnp.asarray(states), params)
        aaae(calculated, expected)


def test_log_ces() -> None:
    states = jnp.array([3, 7.5])
    params = jnp.array([0.4, 0.6, 2])
    expected = 7.244628323025
    calculated = log_ces(states, params)
    aaae(calculated, expected)


def test_where_all_but_one_gammas_are_zero() -> None:
    """This has to be tested, becaus it leads to an underflow in the log step."""
    states = jnp.ones(3)
    params = jnp.array([0, 0, 1, -0.5])
    calculated = log_ces(states, params)
    expected = 1.0
    aaae(calculated, expected)


def test_constant() -> None:
    assert constant("bla", "blubb") == "bla"  # ty: ignore[invalid-argument-type]


def test_robust_translog() -> None:
    all_states = jnp.array(
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

    params = jnp.array(
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
        calculated = robust_translog(jnp.asarray(states), params)
        aaae(calculated, expected)


def test_log_ces_general() -> None:
    states = jnp.array([3, 7.5])
    params = jnp.array([0.4, 0.6, 2, 2, 0.5])
    expected = 7.244628323025
    calculated = log_ces_general(states, params)
    aaae(calculated, expected)


def test_log_ces_general_where_all_but_one_gammas_are_zero() -> None:
    """This has to be tested, becaus it leads to an underflow in the log step."""
    states = jnp.ones(3)
    params = jnp.array([0, 0, 1, -0.5, -0.5, -0.5, -2])
    calculated = log_ces_general(states, params)
    expected = 1.0
    aaae(calculated, expected)


def test_param_names_log_ces_general() -> None:
    factors = ("a", "b")
    expected = ["a", "b", "sigma_a", "sigma_b", "tfp"]
    calculated = params_log_ces_general(factors)
    assert calculated == expected


# --- Tests for params_* functions ---


def test_params_linear() -> None:
    factors = ("a", "b", "c")
    result = params_linear(factors)
    assert result == ["a", "b", "c", "constant"]


def test_params_translog() -> None:
    factors = ("a", "b", "c")
    result = params_translog(factors)
    # 3 linear + 3 squares + 3 interactions + 1 constant = 10
    assert len(result) == 10
    assert result[:3] == ["a", "b", "c"]
    assert result[3:6] == ["a ** 2", "b ** 2", "c ** 2"]
    assert result[6:9] == ["a * b", "a * c", "b * c"]
    assert result[9] == "constant"


def test_params_robust_translog() -> None:
    factors = ("a", "b", "c")
    assert params_robust_translog(factors) == params_translog(factors)


def test_params_linear_and_squares() -> None:
    factors = ("a", "b", "c")
    result = params_linear_and_squares(factors)
    # 3 linear + 3 squares + 1 constant = 7
    assert len(result) == 7
    assert result[:3] == ["a", "b", "c"]
    assert result[3:6] == ["a ** 2", "b ** 2", "c ** 2"]
    assert result[6] == "constant"


def test_params_constant() -> None:
    assert params_constant(("a", "b", "c")) == []


def test_params_log_ces() -> None:
    factors = ("a", "b", "c")
    result = params_log_ces(factors)
    assert result == ["a", "b", "c", "phi"]


# --- Tests for linear_and_squares function ---


def test_linear_and_squares() -> None:
    states = jnp.array([1.0, 2.0, 3.0])
    # 3 linear + 3 square + 1 constant = 7
    params = jnp.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.5])
    expected = 1.4 + 0.36 + 0.5
    aaae(linear_and_squares(states, params), expected)


# --- Tests for identity_constraints_* functions ---


def test_identity_constraints_linear() -> None:
    all_factors = ("a", "b", "c")
    result = identity_constraints_linear("a", 0, all_factors)
    assert len(result) == 4  # 3 factors + constant
    # "a" regressor should be fixed at 1.0
    assert result[0]["value"] == 1.0
    assert result[0]["loc"] == ("transition", 0, "a", "a")
    # others should be 0.0
    assert result[1]["value"] == 0.0
    assert result[3]["value"] == 0.0  # constant


def test_identity_constraints_translog() -> None:
    all_factors = ("a", "b", "c")
    result = identity_constraints_translog("a", 0, all_factors)
    # Should have one constraint per translog param
    assert len(result) == len(params_translog(all_factors))
    # First constraint for "a" linear should be 1.0
    assert result[0]["value"] == 1.0
    # All others should be 0.0
    for c in result[1:]:
        assert c["value"] == 0.0


def test_identity_constraints_robust_translog() -> None:
    all_factors = ("a", "b")
    result_robust = identity_constraints_robust_translog("a", 0, all_factors)
    result_translog = identity_constraints_translog("a", 0, all_factors)
    assert result_robust == result_translog


def test_identity_constraints_linear_and_squares() -> None:
    all_factors = ("a", "b", "c")
    result = identity_constraints_linear_and_squares("a", 0, all_factors)
    assert len(result) == len(params_linear_and_squares(all_factors))
    assert result[0]["value"] == 1.0  # "a" linear
    for c in result[1:]:
        assert c["value"] == 0.0


def test_identity_constraints_log_ces_raises() -> None:
    with pytest.raises(NotImplementedError):
        identity_constraints_log_ces(("a", "b"), 0, ("a", "b"))


def test_identity_constraints_log_ces_general_raises() -> None:
    with pytest.raises(NotImplementedError):
        identity_constraints_log_ces_general(("a", "b"), 0, ("a", "b"))


def test_constraints_log_ces() -> None:
    result = constraints_log_ces("fac1", ("a", "b", "c"), 0)
    assert result["type"] == "probability"
    assert len(result["loc"]) == 3  # gamma constraints for a, b, c (not phi)
    for loc in result["loc"]:
        assert loc[0] == "transition"
        assert loc[1] == 0
        assert loc[2] == "fac1"
