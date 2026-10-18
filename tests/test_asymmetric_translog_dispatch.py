"""Public-model, AMN-resolver and AF-validation dispatch of asymmetric_translog."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from skillmodels.af.validate import (
    _AF_COMPATIBLE_TRANSITIONS,
    _LEAKY_BUILTIN_PRODUCTION,
    validate_af_model,
)
from skillmodels.amn.simulate_and_regress import _resolve_transition_callable
from skillmodels.common import transition_functions as tf
from skillmodels.common.model_spec import FactorSpec, ModelSpec
from skillmodels.common.process_model import process_model
from skillmodels.common.types import Normalizations

jax.config.update("jax_enable_x64", True)

NAMES = ("skill", "skill ** 2 (above)", "skill ** 2 (below)", "skill knot", "constant")
# β = 1, curvature 0.5 below the knot at 0 only: f(2) = 2, f(-2) = -2 + 0.5 * 4 = 0.
PARAMS = jnp.array([1.0, 0.0, 0.5, 0.0, 0.0])


@pytest.fixture
def model_and_processed():
    normalizations = Normalizations(
        loadings=({"m0": 1.0}, {"m0": 1.0}),
        intercepts=({"m0": 0.0}, {"m0": 0.0}),
    )
    factor = FactorSpec(
        measurements=(("m0", "m1", "m2"), ("m0", "m1", "m2")),
        normalizations=normalizations,
        transition_function="asymmetric_translog",
    )
    model = ModelSpec(factors={"skill": factor})
    return model, process_model(model)


def test_public_model_uses_asymmetric_law(model_and_processed) -> None:
    _, processed = model_and_processed
    info = processed.transition_info
    assert info is not None
    states = jnp.array([[2.0], [-2.0]])
    params = {"skill": PARAMS}
    expected = jnp.array([[2.0], [0.0]])
    assert_allclose(info.func(params, states), expected, atol=1e-14)
    assert_allclose(jax.jit(info.func)(params, states), expected, atol=1e-14)
    assert tuple(info.param_names["skill"]) == NAMES


def test_amn_resolver_returns_the_asymmetric_law(model_and_processed) -> None:
    model, processed = model_and_processed
    func, names = _resolve_transition_callable(
        "asymmetric_translog", "skill", processed, model
    )
    assert func is tf.asymmetric_translog
    assert names == NAMES
    assert_allclose(func(jnp.array([2.0]), PARAMS), 2.0, atol=1e-14)
    assert_allclose(func(jnp.array([-2.0]), PARAMS), 0.0, atol=1e-14)


def test_af_allow_lists_and_validation(model_and_processed) -> None:
    assert "asymmetric_translog" in _AF_COMPATIBLE_TRANSITIONS
    assert "asymmetric_translog" in _LEAKY_BUILTIN_PRODUCTION
    model, _ = model_and_processed
    validate_af_model(model)
