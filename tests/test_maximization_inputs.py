"""Tests for maximization input functions."""

import jax.numpy as jnp
import numpy as np

from skillmodels.maximization_inputs import _to_numpy


def test_to_numpy_with_dict() -> None:
    """Test _to_numpy with dictionary input."""
    dict_ = {"a": jnp.ones(3), "b": 4.5}
    calculated = _to_numpy(dict_)
    assert isinstance(calculated["a"], np.ndarray)
    assert isinstance(calculated["b"], float)


def test_to_numpy_one_array() -> None:
    """Test _to_numpy with single array input."""
    calculated = _to_numpy(jnp.ones(3))
    assert isinstance(calculated, np.ndarray)


def test_to_numpy_one_float() -> None:
    """Test _to_numpy with single float input."""
    calculated = _to_numpy(3.5)
    assert isinstance(calculated, float)
