"""Tests for maximization input functions."""

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import (
    _get_jnp_params_vec,
    _to_numpy,
    get_maximization_inputs,
)
from skillmodels.config import TEST_DATA_DIR
from skillmodels.constraints import FixedConstraintWithValue
from skillmodels.test_data.model2 import MODEL2
from skillmodels.utilities import reduce_n_periods


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


@pytest.fixture
def model2_short():
    return reduce_n_periods(MODEL2, new_n_periods=3)


@pytest.fixture
def model2_data():
    return pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta").set_index(
        ["caseid", "period"]
    )


def test_get_maximization_inputs_with_fixed_params_pins_cross_factor_gamma(
    model2_short, model2_data
) -> None:
    """Fix gamma_fac3 in log_ces at 0 via fixed_params; verify CHS pipeline.

    Before probability + fixed-param support in optimagic, combining a
    `ProbabilityConstraint` with a `FixedConstraint` on one of its selected
    entries raised `InvalidConstraintError`. Now the fold machinery removes
    the fixed entry from the selector; CHS should build a valid problem
    whose params_template and constraint list reflect the pin and whose
    log-likelihood evaluates to a finite number.
    """
    fixed_idx = pd.MultiIndex.from_tuples(
        [
            ("transition", 0, "fac1", "fac3"),
            ("transition", 1, "fac1", "fac3"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [0.0, 0.0]}, index=fixed_idx)

    inputs = get_maximization_inputs(model2_short, model2_data, fixed_params=fixed_df)

    template = inputs["params_template"]
    assert template.loc[("transition", 0, "fac1", "fac3"), "value"] == 0.0
    assert template.loc[("transition", 1, "fac1", "fac3"), "value"] == 0.0
    user_fixed = [
        c
        for c in inputs["constraints"]
        if isinstance(c, FixedConstraintWithValue) and c.loc in set(fixed_idx)
    ]
    assert len(user_fixed) == 2

    # optimagic should accept the combined problem with our fold helper.
    params = template.copy()
    # Fill free entries with reasonable starting values compatible with the
    # simplex constraint: split the remaining 1.0 between fac1 and fac2.
    for t in (0, 1):
        params.loc[("transition", t, "fac1", "fac1"), "value"] = 0.5
        params.loc[("transition", t, "fac1", "fac2"), "value"] = 0.5
    params["value"] = params["value"].fillna(0.1)

    om.check_constraints(
        params=params[["value"]],
        constraints=inputs["constraints"],
    )

    loglike_val = inputs["loglike"](params)
    assert np.isfinite(loglike_val)


def test_get_maximization_inputs_with_fixed_params_non_zero(
    model2_short, model2_data
) -> None:
    """Fix a gamma at a non-zero value; remaining simplex sums to 1 - c."""
    fixed_idx = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac3")],
        names=["category", "period", "name1", "name2"],
    )
    fixed_df = pd.DataFrame({"value": [0.2]}, index=fixed_idx)

    inputs = get_maximization_inputs(model2_short, model2_data, fixed_params=fixed_df)

    template = inputs["params_template"]
    assert template.loc[("transition", 0, "fac1", "fac3"), "value"] == 0.2
    params = template.copy()
    params.loc[("transition", 0, "fac1", "fac1"), "value"] = 0.4
    params.loc[("transition", 0, "fac1", "fac2"), "value"] = 0.4
    params.loc[("transition", 1, "fac1", "fac1"), "value"] = 0.4
    params.loc[("transition", 1, "fac1", "fac2"), "value"] = 0.4
    params.loc[("transition", 1, "fac1", "fac3"), "value"] = 0.2
    params["value"] = params["value"].fillna(0.1)

    om.check_constraints(
        params=params[["value"]],
        constraints=inputs["constraints"],
    )

    loglike_val = inputs["loglike"](params)
    assert np.isfinite(loglike_val)
