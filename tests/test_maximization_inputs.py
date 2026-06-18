"""Tests for maximization input functions."""

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import pytest

from skillmodels.chs.maximization_inputs import (
    _fail_if_start_params_incomplete,
    _get_jnp_params_vec,
    _to_numpy,
    get_maximization_inputs,
)
from skillmodels.chs.options import CHSEstimationOptions
from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.constraints import FixedConstraintWithValue
from skillmodels.common.utilities import reduce_n_periods
from skillmodels.test_data.model2 import MODEL2, MODEL2_CHS_OPTIONS


def test_fail_if_start_params_incomplete_flags_non_finite_rows() -> None:
    """A seeded start point with a missing/non-finite value fails early + clearly."""
    names = ["category", "period", "name1", "name2"]
    idx = pd.MultiIndex.from_tuples(
        [("loadings", 0, "y1", "skills"), ("meas_sds", 0, "y1", "-")], names=names
    )

    complete = pd.DataFrame({"value": [1.0, 0.5]}, index=idx)
    _fail_if_start_params_incomplete(complete)  # all finite -> no raise

    incomplete = pd.DataFrame({"value": [1.0, np.nan]}, index=idx)
    with pytest.raises(ValueError, match="without a finite value"):
        _fail_if_start_params_incomplete(incomplete)


@pytest.mark.long_running
def test_amn_start_params_satisfy_equality_constraints() -> None:
    """AMN-seeded start values must satisfy the stage-equality constraints.

    AMN estimates parameters per aug_period, but the stage
    `PairwiseEqualityConstraint`s tie transition / shock params across the
    aug_periods of a stage. The AMN seed must re-pool those groups; otherwise
    `optimagic` rejects the start point with `InvalidParamsError`.
    """
    data = pd.read_stata(TEST_DATA_DIR / "model2_simulated_data.dta").set_index(
        ["caseid", "period"]
    )
    mi = get_maximization_inputs(
        model_spec=MODEL2,
        data=data,
        chs_options=CHSEstimationOptions(start_params_strategy="amn"),
    )
    template = mi["params_template"]
    # optimagic validates start-point feasibility at setup and raises
    # InvalidParamsError on a violated equality constraint. A trivial one-step
    # maximize reaches that check without running the Kalman likelihood.
    res = om.maximize(
        fun=lambda p: float(p["value"].sum()),
        params=template[["value"]],
        algorithm="scipy_lbfgsb",
        bounds=om.Bounds(lower=template["lower_bound"], upper=template["upper_bound"]),
        constraints=mi["constraints"],
        algo_options={"stopping_maxiter": 1},
    )
    assert res is not None


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

    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS, fixed_params=fixed_df
    )

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

    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS, fixed_params=fixed_df
    )

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


def test_get_maximization_inputs_accepts_fixed_params_keyed_by_period(
    model2_short, model2_data
) -> None:
    """`fixed_params` keyed by `period` (the public name) must pin the entry.

    `params_index` uses `aug_period` internally; `MultiIndex.intersection`
    silently returns an empty set when level names differ across the
    operands, so without name alignment the user's pin would vanish.
    Regression for that silent-drop bug.
    """
    fixed_idx = pd.MultiIndex.from_tuples(
        [("transition", 0, "fac1", "fac3")],
        names=["category", "period", "name1", "name2"],  # public-facing name
    )
    fixed_df = pd.DataFrame({"value": [0.2]}, index=fixed_idx)

    inputs = get_maximization_inputs(
        model2_short, model2_data, chs_options=MODEL2_CHS_OPTIONS, fixed_params=fixed_df
    )

    template = inputs["params_template"]
    assert template.loc[("transition", 0, "fac1", "fac3"), "value"] == pytest.approx(
        0.2
    )
    fixed_constraints = [
        c
        for c in inputs["constraints"]
        if isinstance(c, FixedConstraintWithValue)
        and c.loc == ("transition", 0, "fac1", "fac3")
    ]
    assert len(fixed_constraints) == 1
    assert fixed_constraints[0].value == pytest.approx(0.2)
