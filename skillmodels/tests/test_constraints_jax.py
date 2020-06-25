import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.constraints_jax import _anchoring_constraints
from skillmodels.constraints_jax import _constant_factors_constraints
from skillmodels.constraints_jax import _initial_mean_constraints
from skillmodels.constraints_jax import _mixture_weight_constraints
from skillmodels.constraints_jax import _normalization_constraints
from skillmodels.constraints_jax import _not_measured_constraints
from skillmodels.constraints_jax import _stage_constraints
from skillmodels.constraints_jax import _trans_coeff_constraints
from skillmodels.constraints_jax import add_bounds


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


# ======================================================================================
# constraints due to normalizations of loadings and variances
# ======================================================================================


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


# ======================================================================================
# constraints the set the loadings of factors that are not measured to zero
# ======================================================================================


def test_not_measured_constraints():
    ind_tups = [(0, "m1"), (0, "m2"), (0, "m3"), (1, "m1"), (1, "m3")]
    data = [[True, False], [True, True], [False, True], [True, False], [False, True]]
    columns = ["fac1", "fac2"]
    uinfo = pd.DataFrame(
        data, columns=columns, index=pd.MultiIndex.from_tuples(ind_tups)
    )
    labels = {"factors": columns}

    expected = [
        {
            "loc": [
                ("loading", 0, "m1", "fac2"),
                ("loading", 0, "m3", "fac1"),
                ("loading", 1, "m1", "fac2"),
                ("loading", 1, "m3", "fac1"),
            ],
            "type": "fixed",
            "value": 0.0,
        }
    ]

    calculated = _not_measured_constraints(uinfo, labels)
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints for mixture weights
# ======================================================================================


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


# ======================================================================================
# constraints for development stages
# ======================================================================================


def test_stage_constraints():
    stages = [0]
    stagemap = [0] * 3

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

    calculated = _stage_constraints(stagemap, stages)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints for constant factors
# ======================================================================================


def test_constant_factor_constraints():
    labels = {
        "factors": ["fac1", "fac2"],
        "periods": [0, 1, 2],
        "transition_names": ["bla", "constant"],
    }

    expected = [
        {"loc": ("shock_sd", 0, "fac2", "-"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sd", 1, "fac2", "-"), "type": "fixed", "value": 0.0},
    ]

    calculated = _constant_factors_constraints(labels)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints that ensure the ordering of initial states (needed for identification)
# ======================================================================================


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


# ======================================================================================
# constraints on transition parameters
# ======================================================================================


def test_trans_coeff_constraints():
    labels = {
        "factors": ["fac1", "fac2", "fac3"],
        "transition_names": ["log_ces", "bla", "blubb"],
        "periods": [0, 1, 2],
    }

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
    calculated = _trans_coeff_constraints(labels)

    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints on anchoring parameters
# ======================================================================================


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


@pytest.fixture
def base_anchoring_info():
    anch_info = {
        "factors": ["f1", "f2"],
        "outcome": "outcome",
        "use_controls": True,
        "use_constant": True,
        "free_loadings": True,
    }
    return anch_info


def test_anchoring_constraints_no_constraint_needed(anch_uinfo, base_anchoring_info):
    calculated = _anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))
    assert calculated == []


def test_anchoring_constraints_for_constants(anch_uinfo, base_anchoring_info):
    base_anchoring_info["use_constant"] = False
    calculated = _anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))

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


def test_anchoring_constraints_for_controls(anch_uinfo, base_anchoring_info):
    base_anchoring_info["use_controls"] = False
    calculated = _anchoring_constraints(
        anch_uinfo, ["c1", "c2"], base_anchoring_info, (0, 1)
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


def test_anchoring_constraints_for_loadings(anch_uinfo, base_anchoring_info):
    base_anchoring_info["free_loadings"] = False
    calculated = _anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))

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
