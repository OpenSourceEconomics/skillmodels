import jax.numpy as jnp
import numpy as np
import pytest

from skillmodels.maximization_inputs import _to_numpy, model_has_investments


def test_to_numpy_with_dict():
    dict_ = {"a": jnp.ones(3), "b": 4.5}
    calculated = _to_numpy(dict_)
    assert isinstance(calculated["a"], np.ndarray)
    assert isinstance(calculated["b"], float)


def test_to_numpy_one_array():
    calculated = _to_numpy(jnp.ones(3))
    assert isinstance(calculated, np.ndarray)


def test_to_numpy_one_float():
    calculated = _to_numpy(3.5)
    assert isinstance(calculated, float)


def test_model_has_investments_not_specified():
    factors = {"a": {}}
    assert model_has_investments(factors) == False


def test_model_has_investments_wrong_type():
    factors = {"a": {"is_investment": 3}}
    with pytest.raises(ValueError):
        model_has_investments(factors)


def test_model_has_investments_wrong_constellation():
    factors = {"a": {"is_investment": False, "is_correction": True}}
    with pytest.raises(ValueError):
        model_has_investments(factors)


def test_model_has_investments_indeed():
    factors = {
        "a": {"is_investment": True, "is_correction": False},
        "b": {"is_investment": False, "is_correction": False},
    }
    assert model_has_investments(factors) == True


def test_model_has_investments_and_correction():
    factors = {
        "a": {"is_investment": True, "is_correction": False},
        "b": {"is_investment": False, "is_correction": False},
        "c": {"is_investment": True, "is_correction": True},
    }
    assert model_has_investments(factors) == True
