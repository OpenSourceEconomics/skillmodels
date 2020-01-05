from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from nose.tools import assert_equal
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from skillmodels.pre_processing.model_spec_processor import (
    _check_and_clean_normalizations_list,
)
from skillmodels.pre_processing.model_spec_processor import (
    _check_and_fill_normalization_specification,
)
from skillmodels.pre_processing.model_spec_processor import _check_measurements
from skillmodels.pre_processing.model_spec_processor import (
    _clean_controls_specification,
)
from skillmodels.pre_processing.model_spec_processor import _factor_update_info
from skillmodels.pre_processing.model_spec_processor import _purpose_update_info
from skillmodels.pre_processing.model_spec_processor import (
    _transition_equation_included_factors,
)
from skillmodels.pre_processing.model_spec_processor import _transition_equation_names


def test_transition_names():
    model_specs = {}
    model_specs["factors"] = ("f1", "f2", "f3")
    names = ("linear", "ces", "ar1")
    model_specs["_facinf"] = {
        factor: {"trans_eq": {"name": name}}
        for factor, name in zip(model_specs["factors"], names)
    }
    assert_equal(_transition_equation_names(model_specs)["transition_names"], names)


@pytest.fixture
def transeq_setup():
    model_specs = {}
    model_specs["factors"] = ("f1", "f2")
    model_specs["_facinf"] = {
        factor: {"trans_eq": {"included_factors": []}}
        for factor in model_specs["factors"]
    }
    model_specs["_facinf"]["f1"]["trans_eq"]["included_factors"] = ["f2", "f1"]
    model_specs["_facinf"]["f2"]["trans_eq"]["included_factors"] = ["f2"]
    model_specs["nfac"] = 2
    return model_specs


def test_transition_equation_included_factors(transeq_setup):
    included_factors = _transition_equation_included_factors(transeq_setup)
    assert_equal(included_factors["included_factors"], (("f1", "f2"), ("f2",)))


def test_transition_equation_included_factor_positions(transeq_setup):
    positions = _transition_equation_included_factors(transeq_setup)
    exp_positions = [np.array([0, 1]), np.array([1])]
    for pos, exp_pos in zip(positions["included_positions"], exp_positions):
        assert_array_equal(pos, exp_pos)


def test_check_measurements():
    model_specs = {}
    model_specs["periods"] = (0, 1)
    inf = {"f1": {}, "f2": {}}
    inf["f1"]["measurements"] = [["m1", "m2", "m3", "m4"]] * 2
    inf["f2"]["measurements"] = [["m5", "m6", "m7", "m8"]] * 2
    model_specs["_facinf"] = inf
    model_specs["factors"] = tuple(sorted(model_specs["_facinf"].keys()))
    model_specs["transition_names"] = ("log_ces", "blubb")
    cols = ["__period__"] + [f"m{i}" for i in range(1, 9)]
    dat = np.vstack([np.zeros(9), np.ones(9)])
    model_specs["data"] = pd.DataFrame(columns=cols, data=dat)
    res = {}
    res["f1"] = model_specs["_facinf"]["f1"]["measurements"]
    res["f2"] = model_specs["_facinf"]["f2"]["measurements"]
    checked_meas = _check_measurements(model_specs)["measurements"]
    assert_equal(checked_meas, res)


def test_clean_control_specs_nothing_to_clean():
    model_specs = {}
    model_specs["_timeinf"] = {"controls": [["c1", "c2"], ["c1", "c2"]]}
    model_specs["periods"] = [0, 1]
    model_specs["nperiods"] = 2
    model_specs["model_name"] = "model"
    model_specs["dataset_name"] = "data"
    cols = ["__period__", "c1", "c2", "m1", "m2"]
    dat = np.zeros((10, 5))
    dat[5:, 0] = 1
    model_specs["data"] = DataFrame(data=dat, columns=cols)
    model_specs["measurements"] = {"f1": [["m1", "m2"]] * 2, "f2": [["m1", "m2"]] * 2}
    model_specs["factors"] = ["f1", "f2"]
    res = (("c1", "c2"), ("c1", "c2"))
    cont_spec = _clean_controls_specification(model_specs)
    assert_equal(cont_spec["controls"], res)


@pytest.fixture
def norm_list_setup():
    model_specs = {}
    model_specs["periods"] = [0, 1]
    model_specs["_facinf"] = {"f1": {"measurements": [["m1", "m2", "m3", "m4"]] * 2}}
    model_specs["f1_norm_list"] = [{"m1": 1}, {"m1": 1}]
    model_specs["factors"] = sorted(model_specs["_facinf"].keys())
    model_specs["measurements"] = {"f1": [["m1", "m2", "m3", "m4"]] * 2}
    model_specs["model_name"] = "model"
    model_specs["dataset_name"] = "data"
    model_specs["nperiods"] = len(model_specs["periods"])
    return model_specs


def test_check_normalizations_no_error_dictionaries(norm_list_setup):
    result = _check_and_clean_normalizations_list(
        norm_list_setup, "f1", norm_list_setup["f1_norm_list"], "loadings"
    )
    assert_equal(result, [{"m1": 1}, {"m1": 1}])


def test_check_normalizations_not_specified_error(norm_list_setup):
    f1_norm_list = [{"m10": 1}, {"m1": 1}]
    assert_raises(
        KeyError,
        _check_and_clean_normalizations_list,
        norm_list_setup,
        "f1",
        f1_norm_list,
        "loadings",
    )


def test_check_normalizations_invalid_length_of_list(norm_list_setup):
    assert_raises(
        AssertionError,
        _check_and_clean_normalizations_list,
        norm_list_setup,
        "f1",
        norm_list_setup["f1_norm_list"] * 2,
        "loadings",
    )


def test_check_normalizations_invalid_length_of_sublist(norm_list_setup):
    f1_norm_list = [["m1"], ["m1", 1]]
    assert_raises(
        AssertionError,
        _check_and_clean_normalizations_list,
        norm_list_setup,
        "f1",
        f1_norm_list,
        "loadings",
    )


@patch(
    "skillmodels.pre_processing.model_spec_processor."
    + "_check_and_clean_normalizations_list",
    return_value=[{"a": 1}, {"a": 1}, {"a": 1}],
    autospec=True,
)
def test_check_and_fill_normalization_specifications(mock_check_and_clean_norm_list):
    res = {
        "f1": {
            "loadings": [{"a": 1}, {"a": 1}, {"a": 1}],
            "intercepts": [{"a": 1}, {"a": 1}, {"a": 1}],
        },
        "f2": {"loadings": [{}, {}, {}], "intercepts": [{}, {}, {}]},
    }

    model_specs = {}
    model_specs["_facinf"] = {
        "f1": {
            "normalizations": {
                "loadings": [{"a": 1}, {"a": 1}, {"a": 1}],
                "intercepts": [{"a": 1}, {"a": 1}, {"a": 1}],
            }
        },
        "f2": {},
    }
    model_specs["nperiods"] = 3
    model_specs["factors"] = sorted(model_specs["_facinf"].keys())
    norm_specs = _check_and_fill_normalization_specification(model_specs)
    assert_equal(norm_specs["normalizations"], res)


def test_check_and_fill_normalization_specifications_raise_exception():
    model_spec = {}
    model_spec["_facinf"] = {
        "f1": {
            "normalizations": {
                "h": [{"a": 1}, {"a": 1}, {"a": 1}],
                "intercepts": [{"a": 1}, {"a": 1}, {"a": 1}],
            }
        }
    }
    model_spec["nperiods"] = 3
    model_spec["factors"] = sorted(model_spec["_facinf"].keys())
    assert_raises(ValueError, _check_and_fill_normalization_specification, model_spec)


def factor_uinfo():

    dat = [
        [0, "m1", 1, 0, 0],
        [0, "m2", 1, 0, 0],
        [0, "m3", 0, 1, 0],
        [0, "m4", 0, 1, 0],
        [0, "m5", 0, 1, 1],
        [0, "m6", 0, 0, 1],
        [0, "a_f1", 1, 0, 0],
        [1, "m1", 1, 0, 0],
        [1, "m2", 1, 0, 0],
        [1, "m3", 0, 1, 0],
        [1, "m4", 0, 1, 0],
        [1, "m5", 0, 1, 1],
        [1, "m6", 0, 0, 1],
        [1, "a_f1", 1, 0, 0],
        [2, "m1", 1, 0, 0],
        [2, "m2", 1, 0, 0],
        [2, "m3", 0, 1, 0],
        [2, "m4", 0, 1, 0],
        [2, "m5", 0, 1, 1],
        [2, "m6", 0, 0, 1],
        [2, "a_f1", 1, 0, 0],
        [3, "m1", 1, 0, 0],
        [3, "m2", 1, 0, 0],
        [3, "m3", 0, 1, 0],
        [3, "m4", 0, 1, 0],
        [3, "m5", 0, 1, 1],
        [3, "m6", 0, 0, 1],
        [3, "a_f1", 1, 0, 0],
    ]
    cols = ["period", "variable", "f1", "f2", "f3"]
    df = DataFrame(data=dat, columns=cols)
    df.set_index(["period", "variable"], inplace=True)
    return df


def test_factor_update_info():
    model_spec = {}
    model_spec["periods"] = [0, 1, 2, 3]
    model_spec["nperiods"] = len(model_spec["periods"])
    model_spec["factors"] = ["f1", "f2", "f3"]
    model_spec["nfac"] = len(model_spec["factors"])
    model_spec["measurements"] = {
        "f1": [["m1", "m2"]] * 4,
        "f2": [["m3", "m4", "m5"]] * 4,
        "f3": [["m5", "m6"]] * 4,
    }
    model_spec["anch_outcome"] = "a"
    model_spec["anchored_factors"] = ["f1"]
    model_spec["anchoring"] = True

    calc = _factor_update_info(model_spec)
    exp = factor_uinfo()
    assert_frame_equal(calc, exp, check_dtype=False)


def purpose_uinfo():
    ind = factor_uinfo().index
    dat = (["measurement"] * 6 + ["anchoring"]) * 4
    return pd.Series(index=ind, data=dat, name="purpose")


@patch(
    "skillmodels.pre_processing.model_spec_processor._factor_update_info",
    return_value=factor_uinfo(),
    autospec=True,
)
def test_update_info_purpose(mock_factor_update_info):
    model_specs = {}
    model_specs["anchoring"] = True
    model_specs["anchored_factors"] = ["f1"]
    model_specs["anch_outcome"] = "a"
    model_specs["nperiods"] = 4
    model_specs["periods"] = (0, 1, 2, 3)
    calc = _purpose_update_info(model_specs)
    exp = purpose_uinfo()
    assert calc.equals(exp)
