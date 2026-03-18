"""Tests for maximization input functions."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from skillmodels.maximization_inputs import _get_jnp_params_vec, _to_numpy


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


def test_get_jnp_params_vec_missing_entries_raises() -> None:
    target_index = pd.MultiIndex.from_tuples(
        [("a", 0, "x", "y"), ("b", 0, "x", "y")],
        names=["category", "period", "name1", "name2"],
    )
    # Params has only one of the two entries
    params = pd.DataFrame(
        {"value": [1.0]},
        index=target_index[:1],
    )
    with pytest.raises(ValueError, match="missing entries"):
        _get_jnp_params_vec(params, target_index)


def test_get_jnp_params_vec_additional_entries_raises() -> None:
    target_index = pd.MultiIndex.from_tuples(
        [("a", 0, "x", "y")],
        names=["category", "period", "name1", "name2"],
    )
    params = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0, "x", "y"), ("extra", 0, "x", "y")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    with pytest.raises(ValueError, match="additional entries"):
        _get_jnp_params_vec(params, target_index)
