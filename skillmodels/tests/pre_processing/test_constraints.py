import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.pre_processing.constraints import _anchoring_constraints
from skillmodels.pre_processing.constraints import _constant_factors_constraints
from skillmodels.pre_processing.constraints import _initial_cov_constraints
from skillmodels.pre_processing.constraints import _initial_mean_constraints
from skillmodels.pre_processing.constraints import _invariant_meas_system_constraints
from skillmodels.pre_processing.constraints import _mixture_weight_constraints
from skillmodels.pre_processing.constraints import _normalization_constraints
from skillmodels.pre_processing.constraints import _not_measured_constraints
from skillmodels.pre_processing.constraints import _stage_constraints
from skillmodels.pre_processing.constraints import _trans_coeff_constraints
from skillmodels.pre_processing.constraints import add_bounds


def test_add_bounds():
    ind_tups = [("shock_sd", i) for i in range(5)] + [
        ("meas_sd", 4),
        ("bla", "blubb"),
        ("meas_sd", "foo"),
    ]
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(ind_tups),
        data=np.arange(16).reshape(8, 2),
        columns=["lower", "upper"],
    )
    expected = df.copy(deep=True)
    expected["lower"] = [0.1] * 5 + [0.1, 12, 0.1]

    calculated = add_bounds(df, 0.1)
    assert_frame_equal(calculated, expected)


def test_invariant_meas_system_constraints():
    ind_tups = [(0, "m1"), (0, "m2"), (1, "m1"), (1, "m3"), (2, "m1"), (2, "m3")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(ind_tups))
    uinfo["is_repeated"] = [False, False, True, False, True, True]
    uinfo["first_occurence"] = [np.nan, np.nan, 0, np.nan, 0, 1]
    controls = [["cont"]] * 3
    factors = ["fac1", "fac2"]

    expected = [
        {
            "loc": [
                ("control_coeffs", 1, "m1", "constant"),
                ("control_coeffs", 0, "m1", "constant"),
            ],
            "type": "equality",
        },
        {
            "loc": [
                ("control_coeffs", 1, "m1", "cont"),
                ("control_coeffs", 0, "m1", "cont"),
            ],
            "type": "equality",
        },
        {
            "loc": [("loading", 1, "m1", "fac1"), ("loading", 0, "m1", "fac1")],
            "type": "equality",
        },
        {
            "loc": [("loading", 1, "m1", "fac2"), ("loading", 0, "m1", "fac2")],
            "type": "equality",
        },
        {
            "loc": [("meas_sd", 1, "m1", "-"), ("meas_sd", 0, "m1", "-")],
            "type": "equality",
        },
        {
            "loc": [
                ("control_coeffs", 2, "m1", "constant"),
                ("control_coeffs", 0, "m1", "constant"),
            ],
            "type": "equality",
        },
        {
            "loc": [
                ("control_coeffs", 2, "m1", "cont"),
                ("control_coeffs", 0, "m1", "cont"),
            ],
            "type": "equality",
        },
        {
            "loc": [("loading", 2, "m1", "fac1"), ("loading", 0, "m1", "fac1")],
            "type": "equality",
        },
        {
            "loc": [("loading", 2, "m1", "fac2"), ("loading", 0, "m1", "fac2")],
            "type": "equality",
        },
        {
            "loc": [("meas_sd", 2, "m1", "-"), ("meas_sd", 0, "m1", "-")],
            "type": "equality",
        },
        {
            "loc": [
                ("control_coeffs", 2, "m3", "constant"),
                ("control_coeffs", 1, "m3", "constant"),
            ],
            "type": "equality",
        },
        {
            "loc": [
                ("control_coeffs", 2, "m3", "cont"),
                ("control_coeffs", 1, "m3", "cont"),
            ],
            "type": "equality",
        },
        {
            "loc": [("loading", 2, "m3", "fac1"), ("loading", 1, "m3", "fac1")],
            "type": "equality",
        },
        {
            "loc": [("loading", 2, "m3", "fac2"), ("loading", 1, "m3", "fac2")],
            "type": "equality",
        },
        {
            "loc": [("meas_sd", 2, "m3", "-"), ("meas_sd", 1, "m3", "-")],
            "type": "equality",
        },
    ]

    calculated = _invariant_meas_system_constraints(uinfo, controls, factors)
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


def test_normalization_constraints():
    norm = {
        "fac1": {
            "loadings": [{"m1": 2, "m2": 1.5}, {"m1": 3}],
            "intercepts": [{"m1": 0.5}, {}],
        },
        "fac2": {"loadings": [{"m3": 1}, {}], "intercepts": [{}, {}]},
    }

    expected = [
        {"loc": ("loading", 0, "m1", "fac1"), "type": "fixed", "value": 2},
        {"loc": ("loading", 0, "m2", "fac1"), "type": "fixed", "value": 1.5},
        {"loc": ("loading", 1, "m1", "fac1"), "type": "fixed", "value": 3},
        {"loc": ("control_coeffs", 0, "m1", "constant"), "type": "fixed", "value": 0.5},
        {"loc": ("loading", 0, "m3", "fac2"), "type": "fixed", "value": 1},
    ]

    calculated = _normalization_constraints(norm)
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


def test_not_measured_constraints():
    ind_tups = [(0, "m1"), (0, "m2"), (0, "m3"), (1, "m1"), (1, "m3")]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(ind_tups))

    measurements = {"fac1": [["m1", "m2"], ["m1"]], "fac2": [["m2", "m3"], ["m3"]]}

    expected = [
        {
            "loc": [
                ("loading", 0, "m3", "fac1"),
                ("loading", 0, "m1", "fac2"),
                ("loading", 1, "m3", "fac1"),
                ("loading", 1, "m1", "fac2"),
            ],
            "type": "fixed",
            "value": 0.0,
        }
    ]

    calculated = _not_measured_constraints(uinfo, measurements, [], None)
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


def test_mixture_weight_constraints_mixture():
    calculated = _mixture_weight_constraints(nmixtures=2)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weight", "type": "probability"}]
    assert_list_equal_except_for_order(calculated, expected)


def test_mixture_weight_constraints_normal():
    calculated = _mixture_weight_constraints(nmixtures=1)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weight", "type": "fixed", "value": 1.0}]
    assert_list_equal_except_for_order(calculated, expected)


def test_initial_cov_constraints():
    nmixtures = 2
    expected = [
        {
            "loc": ("initial_cov", 0, "mixture_0"),
            "type": "covariance",
            "bounds_distance": 0.0,
        },
        {
            "loc": ("initial_cov", 0, "mixture_1"),
            "type": "covariance",
            "bounds_distance": 0.0,
        },
    ]

    calculated = _initial_cov_constraints(nmixtures, 0.0)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_stage_constraints():
    factors = ["fac1"]
    stagemap = [0, 0, 0, 0]
    transition_names = ["linear"]
    included_factors = [["fac1"]]

    expected = [
        {
            "locs": [("trans", 0), ("trans", 1), ("trans", 2)],
            "type": "pairwise_equality",
        },
        {
            "locs": [("shock_sd", 0), ("shock_sd", 1), ("shock_sd", 2)],
            "type": "pairwise_equality",
        },
    ]

    calculated = _stage_constraints(
        stagemap, factors, transition_names, included_factors
    )
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_constant_factor_constraints():
    factors = ["fac1", "fac2"]
    periods = [0, 1, 2]
    transition_names = ["bla", "constant"]

    expected = [
        {"loc": ("shock_sd", 0, "fac2", "-"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sd", 1, "fac2", "-"), "type": "fixed", "value": 0.0},
    ]

    calculated = _constant_factors_constraints(factors, transition_names, periods)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_initial_mean_constraints():
    nmixtures = 3
    factors = ["fac1", "fac2", "fac3"]
    ind_tups = [
        ("initial_mean", 0, "mixture_0", "fac1"),
        ("initial_mean", 0, "mixture_1", "fac1"),
        ("initial_mean", 0, "mixture_2", "fac1"),
    ]

    expected = [{"loc": ind_tups, "type": "increasing"}]

    calculated = _initial_mean_constraints(nmixtures, factors)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_trans_coeff_constraints():
    factors = ["fac1", "fac2", "fac3"]
    transition_names = ["log_ces", "bla", "blubb"]
    included_factors = [["fac1", "fac2", "fac3"], [], []]
    periods = [0, 1, 2]

    expected = [
        {
            "loc": [
                ("trans", 0, "fac1", "fac1"),
                ("trans", 0, "fac1", "fac2"),
                ("trans", 0, "fac1", "fac3"),
            ],
            "type": "probability",
        },
        {
            "loc": [
                ("trans", 1, "fac1", "fac1"),
                ("trans", 1, "fac1", "fac2"),
                ("trans", 1, "fac1", "fac3"),
            ],
            "type": "probability",
        },
    ]
    calculated = _trans_coeff_constraints(
        factors, transition_names, included_factors, periods
    )
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


@pytest.fixture
def anch_uinfo():
    ind_tups = [
        (0, "outcome_f1"),
        (0, "outcome_f2"),
        (0, "m1"),
        (1, "outcome_f1"),
        (1, "outcome_f2"),
        (1, "m1"),
    ]
    uinfo = pd.DataFrame(index=pd.MultiIndex.from_tuples(ind_tups))
    uinfo["purpose"] = ["anchoring", "anchoring", "measurement"] * 2
    return uinfo


def test_anchoring_constraints_no_constraint_needed(anch_uinfo):
    calculated = _anchoring_constraints(
        anch_uinfo, [], "outcome", ["f1", "f2"], True, True, True, (0, 1)
    )
    assert calculated == []


def test_anchoring_constraints_for_constants(anch_uinfo):
    calculated = _anchoring_constraints(
        anch_uinfo, [], "outcome", ["f1", "f2"], True, False, True, (0, 1)
    )

    expected = [
        {
            "loc": ("control_coeffs", 0, "outcome_f1", "constant"),
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": ("control_coeffs", 0, "outcome_f2", "constant"),
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": ("control_coeffs", 1, "outcome_f1", "constant"),
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": ("control_coeffs", 1, "outcome_f2", "constant"),
            "type": "fixed",
            "value": 0,
        },
    ]

    assert calculated == expected


def test_anchoring_constraints_for_controls(anch_uinfo):
    calculated = _anchoring_constraints(
        anch_uinfo, ["c1", "c2"], "outcome", ["f1", "f2"], False, True, True, (0, 1)
    )
    expected = [
        {
            "loc": [
                ("control_coeffs", 0, "outcome_f1", "c1"),
                ("control_coeffs", 0, "outcome_f1", "c2"),
            ],
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": [
                ("control_coeffs", 0, "outcome_f2", "c1"),
                ("control_coeffs", 0, "outcome_f2", "c2"),
            ],
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": [
                ("control_coeffs", 1, "outcome_f1", "c1"),
                ("control_coeffs", 1, "outcome_f1", "c2"),
            ],
            "type": "fixed",
            "value": 0,
        },
        {
            "loc": [
                ("control_coeffs", 1, "outcome_f2", "c1"),
                ("control_coeffs", 1, "outcome_f2", "c2"),
            ],
            "type": "fixed",
            "value": 0,
        },
    ]

    assert calculated == expected


def test_anchoring_constraints_for_loadings(anch_uinfo):
    calculated = _anchoring_constraints(
        anch_uinfo, [], "outcome", ["f1", "f2"], True, True, False, (0, 1)
    )

    expected = [
        {
            "loc": [
                ("loading", 0, "outcome_f1", "f1"),
                ("loading", 0, "outcome_f2", "f2"),
            ],
            "type": "fixed",
            "value": 1,
        },
        {
            "loc": [
                ("loading", 1, "outcome_f1", "f1"),
                ("loading", 1, "outcome_f2", "f2"),
            ],
            "type": "fixed",
            "value": 1,
        },
    ]

    assert calculated == expected


# ======================================================================================


def assert_list_equal_except_for_order(list1, list2):
    for item in list1:
        assert item in list2, f"{item} is in list1 but not in list2"
    for item in list2:
        assert item in list1, f"{item} is in list2 but not in list1"
