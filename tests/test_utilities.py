"""Test utility functions.

All tests should not only assert that modified model specifications are correct but
also that there are no side effects on the inputs.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal, assert_index_equal

from skillmodels.process_model import process_model
from skillmodels.utilities import (
    _get_params_index_from_model_dict,
    _remove_from_dict,
    _remove_from_list,
    _shorten_if_necessary,
    extract_factors,
    reduce_n_periods,
    remove_controls,
    remove_factors,
    remove_measurements,
    switch_linear_to_translog,
    switch_translog_to_linear,
    update_parameter_values,
)

# importing the TEST_DIR from config does not work for test run in conda build
TEST_DIR = Path(__file__).parent.resolve()


@pytest.fixture
def model2():
    with open(TEST_DIR / "model2.yaml") as y:
        model_dict = yaml.load(y, Loader=yaml.FullLoader)
    return model_dict


@pytest.mark.parametrize("factors", ["fac2", ["fac2"]])
def test_extract_factors_single(model2, factors):
    reduced = extract_factors(factors, model2)
    assert list(reduced["factors"]) == ["fac2"]
    assert list(model2["factors"]) == ["fac1", "fac2", "fac3"]
    assert "anchoring" not in reduced
    assert model2["anchoring"]["outcomes"] == {"fac1": "Q1"}
    process_model(reduced)


def test_update_parameter_values():
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
def test_remove_factors(model2, factors):
    reduced = remove_factors(factors, model2)
    assert list(reduced["factors"]) == ["fac1", "fac3"]
    assert list(model2["factors"]) == ["fac1", "fac2", "fac3"]
    assert "anchoring" in reduced
    process_model(reduced)


@pytest.mark.parametrize("measurements", ["y5", ["y5"]])
def test_remove_measurements(model2, measurements):
    reduced = remove_measurements(measurements, model2)
    assert reduced["factors"]["fac2"]["measurements"] == [["y4", "y6"]] * 8
    assert "y5" in model2["factors"]["fac2"]["measurements"][0]
    process_model(reduced)


@pytest.mark.parametrize("controls", ["x1", ["x1"]])
def test_remove_controls(model2, controls):
    reduced = remove_controls(controls, model2)
    assert "controls" not in reduced
    assert "controls" in model2
    process_model(reduced)


def test_reduce_n_periods(model2):
    reduced = reduce_n_periods(model2, 1)
    assert reduced["factors"]["fac1"]["measurements"] == [["y1", "y2", "y3"]]
    assert reduced["factors"]["fac2"]["normalizations"]["loadings"] == [{"y4": 1}]
    process_model(reduced)


def test_switch_linear_to_translog(model2):
    switched = switch_linear_to_translog(model2)
    assert switched["factors"]["fac2"]["transition_function"] == "translog"


def test_switch_linear_and_translog_back_and_forth(model2):
    with_translog = switch_linear_to_translog(model2)
    with_linear = switch_translog_to_linear(with_translog)
    assert model2 == with_linear


@pytest.mark.parametrize("to_remove", ["a", ["a"]])
def test_remove_from_list(to_remove):
    list_ = ["a", "b", "c"]
    calculated = _remove_from_list(list_, to_remove)
    assert calculated == ["b", "c"]
    assert list_ == ["a", "b", "c"]


@pytest.mark.parametrize("to_remove", ["a", ["a"]])
def test_remove_from_dict(to_remove):
    dict_ = {"a": 1, "b": 2, "c": 3}
    calculated = _remove_from_dict(dict_, to_remove)
    assert calculated == {"b": 2, "c": 3}
    assert dict_ == {"a": 1, "b": 2, "c": 3}


def test_reduce_params_via_extract_factors(model2):
    model_dict = reduce_n_periods(model2, 2)

    full_index = _get_params_index_from_model_dict(model_dict)
    params = pd.DataFrame(columns=["value"], index=full_index)

    _, reduced_params = extract_factors("fac3", model_dict, params)

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
            ("initial_states", 0, "mixture_0", "fac3"),
            ("mixture_weights", 0, "mixture_0", "-"),
            ("initial_cholcovs", 0, "mixture_0", "fac3-fac3"),
        ],
        names=["category", "period", "name1", "name2"],
    )

    assert_index_equal(reduced_params.index, expected_index)


def test_extend_params_via_switch_to_translog(model2):
    model_dict = reduce_n_periods(model2, 2)
    normal_index = _get_params_index_from_model_dict(model_dict)
    params = pd.DataFrame(columns=["value"], index=normal_index)

    _, extended_params = switch_linear_to_translog(model_dict, params)

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
        names=["category", "period", "name1", "name2"],
    )

    assert_index_equal(added_index, expected_added_index)

    assert extended_params.loc[added_index, "value"].unique()[0] == 0.05


def test_shorten_if_necessary():
    list_ = list(range(3))
    not_necessary = _shorten_if_necessary(list_, 5)
    assert not_necessary == list_

    necessary = _shorten_if_necessary(list_, 2)
    assert necessary == [0, 1]
