"""Test utility functions.

All tests should not only assert that modified model specifications are correct but
also that there are no side effects on the inputs.

"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from skillmodels.model_spec import ModelSpec
from skillmodels.process_model import process_model
from skillmodels.test_data.model2 import MODEL2
from skillmodels.utilities import (
    _extend_params,
    _get_params_index,
    extract_factors,
    reduce_n_periods,
    remove_controls,
    remove_factors,
    remove_measurements,
    switch_linear_to_translog,
    switch_translog_to_linear,
    update_parameter_values,
)


@pytest.fixture
def model2():
    return MODEL2


@pytest.mark.parametrize("factors", ["fac2", ["fac2"]])
def test_extract_factors_single(model2, factors) -> None:
    reduced = extract_factors(factors, model2)
    assert isinstance(reduced, ModelSpec)
    assert list(reduced.factors) == ["fac2"]
    assert list(model2.factors) == ["fac1", "fac2", "fac3"]
    assert reduced.anchoring is None
    assert model2.anchoring is not None
    assert dict(model2.anchoring.outcomes) == {"fac1": "Q1"}
    process_model(reduced)


def test_update_parameter_values() -> None:
    params = pd.DataFrame()
    params["value"] = np.arange(5, dtype=np.int64)

    others = [
        pd.DataFrame([[7], [8]], columns=["value"], index=[1, 4]),
        pd.DataFrame([[9]], columns=["value"], index=[2]),
    ]

    expected = pd.DataFrame()
    expected["value"] = [0, 7, 9, 3, 8]

    calculated = update_parameter_values(params, others)
    assert_frame_equal(calculated, expected)


@pytest.mark.parametrize("factors", ["fac2", ["fac2"]])
def test_remove_factors(model2, factors) -> None:
    reduced = remove_factors(factors, model2)
    assert isinstance(reduced, ModelSpec)
    assert list(reduced.factors) == ["fac1", "fac3"]
    assert list(model2.factors) == ["fac1", "fac2", "fac3"]
    assert reduced.anchoring is not None
    process_model(reduced)


@pytest.mark.parametrize("measurements", ["y5", ["y5"]])
def test_remove_measurements(model2, measurements) -> None:
    reduced = remove_measurements(measurements, model2)
    assert isinstance(reduced, ModelSpec)
    for period_meas in reduced.factors["fac2"].measurements:
        assert list(period_meas) == ["y4", "y6"]
    assert "y5" in model2.factors["fac2"].measurements[0]
    process_model(reduced)


@pytest.mark.parametrize("controls", ["x1", ["x1"]])
def test_remove_controls(model2, controls) -> None:
    reduced = remove_controls(controls, model2)
    assert isinstance(reduced, ModelSpec)
    assert reduced.controls == ()
    assert model2.controls == ("x1",)
    process_model(reduced)


def test_reduce_n_periods(model2) -> None:
    reduced = reduce_n_periods(model2, 1)
    assert isinstance(reduced, ModelSpec)
    assert list(reduced.factors["fac1"].measurements[0]) == ["y1", "y2", "y3"]
    assert len(reduced.factors["fac1"].measurements) == 1
    norms = reduced.factors["fac2"].normalizations
    assert norms is not None
    assert dict(norms.loadings[0]) == {"y4": 1}
    assert len(norms.loadings) == 1
    process_model(reduced)


def test_switch_linear_to_translog(model2) -> None:
    switched = switch_linear_to_translog(model2)
    assert isinstance(switched, ModelSpec)
    assert switched.factors["fac2"].transition_function == "translog"


def test_switch_linear_and_translog_back_and_forth(model2) -> None:
    with_translog = switch_linear_to_translog(model2)
    assert isinstance(with_translog, ModelSpec)
    with_linear = switch_translog_to_linear(with_translog)
    assert isinstance(with_linear, ModelSpec)
    # Check equivalence of factors
    for name in model2.factors:
        orig = model2.factors[name]
        back = with_linear.factors[name]
        assert orig.measurements == back.measurements
        assert orig.normalizations == back.normalizations
        assert orig.transition_function == back.transition_function
        assert orig.is_endogenous == back.is_endogenous
        assert orig.is_correction == back.is_correction


def test_reduce_params_via_extract_factors(model2) -> None:
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)

    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)

    result = extract_factors("fac3", model, params)
    assert not isinstance(result, ModelSpec)
    _, reduced_params = result

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("controls", 0, "y7", "constant"),
            ("controls", 0, "y7", "x1"),
            ("controls", 0, "y8", "constant"),
            ("controls", 0, "y8", "x1"),
            ("controls", 0, "y9", "constant"),
            ("controls", 0, "y9", "x1"),
            ("loadings", 0, "y7", "fac3"),
            ("loadings", 0, "y8", "fac3"),
            ("loadings", 0, "y9", "fac3"),
            ("meas_sds", 0, "y7", "-"),
            ("meas_sds", 0, "y8", "-"),
            ("meas_sds", 0, "y9", "-"),
            ("shock_sds", 0, "fac3", "-"),
            ("initial_states", 0, "mixture_0", "fac3"),
            ("mixture_weights", 0, "mixture_0", "-"),
            ("initial_cholcovs", 0, "mixture_0", "fac3-fac3"),
        ],
        names=["category", "aug_period", "name1", "name2"],
    )

    assert_index_equal(reduced_params.index, expected_index)


def test_extend_params_via_switch_to_translog(model2) -> None:
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)
    normal_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=normal_index)

    result = switch_linear_to_translog(model, params)
    assert not isinstance(result, ModelSpec)
    _, extended_params = result

    added_index = extended_params.index.difference(normal_index)

    expected_added_index = pd.MultiIndex.from_tuples(
        [
            ("transition", 0, "fac2", "fac1 * fac2"),
            ("transition", 0, "fac2", "fac1 * fac3"),
            ("transition", 0, "fac2", "fac1 ** 2"),
            ("transition", 0, "fac2", "fac2 * fac3"),
            ("transition", 0, "fac2", "fac2 ** 2"),
            ("transition", 0, "fac2", "fac3 ** 2"),
        ],
        names=["category", "aug_period", "name1", "name2"],
    )

    assert_index_equal(added_index, expected_added_index)

    assert extended_params.loc[added_index, "value"].unique()[0] == 0.05


def test_update_parameter_values_single_df() -> None:
    """Pass a single DataFrame instead of a list."""
    params = pd.DataFrame()
    params["value"] = np.arange(5, dtype=np.int64)

    other = pd.DataFrame([[7], [8]], columns=["value"], index=[1, 4])

    expected = pd.DataFrame()
    expected["value"] = [0, 7, 2, 3, 8]

    calculated = update_parameter_values(params, other)
    assert_frame_equal(calculated, expected)


def test_remove_measurements_with_params(model2) -> None:
    """Remove measurements with params and verify tuple return."""
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)
    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)
    params["value"] = 0.1

    result = remove_measurements("y5", model, params)
    assert not isinstance(result, ModelSpec)
    reduced_model, reduced_params = result
    # y5 should not appear in any factor's measurements
    for fspec in reduced_model.factors.values():
        for period_meas in fspec.measurements:
            assert "y5" not in period_meas
    assert isinstance(reduced_params, pd.DataFrame)


def test_remove_controls_with_params(model2) -> None:
    """Remove controls with params and verify tuple return."""
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)
    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)
    params["value"] = 0.1

    result = remove_controls("x1", model, params)
    assert not isinstance(result, ModelSpec)
    reduced_model, reduced_params = result
    assert reduced_model.controls == ()
    assert isinstance(reduced_params, pd.DataFrame)


def test_remove_measurements_warns_on_normalized(model2) -> None:
    with pytest.warns(UserWarning, match="normalized"):
        remove_measurements("y1", model2)


def test_reduce_n_periods_with_params(model2) -> None:
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)
    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)
    params["value"] = 0.1

    result = reduce_n_periods(model, 1, params)
    assert not isinstance(result, ModelSpec)
    _, reduced_params = result
    assert len(reduced_params) < len(params)


def test_extend_params_with_bounds(model2) -> None:
    model = reduce_n_periods(model2, 2)
    assert isinstance(model, ModelSpec)
    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)
    params["value"] = 0.1
    params["lower_bound"] = -1.0
    params["upper_bound"] = 1.0

    translog = switch_linear_to_translog(model)
    assert isinstance(translog, ModelSpec)
    result = _extend_params(params=params, model_spec=translog, fill_value=0.05)
    assert "lower_bound" in result.columns
    assert "upper_bound" in result.columns
    # New entries should have default bounds
    new_rows = result.index.difference(full_index)
    assert (result.loc[new_rows, "lower_bound"] == -np.inf).all()
    assert (result.loc[new_rows, "upper_bound"] == np.inf).all()


def test_switch_translog_to_linear_with_params(model2) -> None:
    """Switch translog to linear with params."""
    # First switch to translog, then back to linear with params
    with_translog = switch_linear_to_translog(model2)
    assert isinstance(with_translog, ModelSpec)

    model = reduce_n_periods(with_translog, 2)
    assert isinstance(model, ModelSpec)
    full_index = _get_params_index(model)
    params = pd.DataFrame(columns=["value"], index=full_index)
    params["value"] = 0.1

    result = switch_translog_to_linear(model, params)
    assert not isinstance(result, ModelSpec)
    reduced_model, reduced_params = result
    assert reduced_model.factors["fac2"].transition_function == "linear"
    assert isinstance(reduced_params, pd.DataFrame)
