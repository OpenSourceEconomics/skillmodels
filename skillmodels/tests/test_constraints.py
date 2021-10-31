import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.constraints import _get_anchoring_constraints
from skillmodels.constraints import _get_constant_factors_constraints
from skillmodels.constraints import _get_initial_states_constraints
from skillmodels.constraints import _get_mixture_weights_constraints
from skillmodels.constraints import _get_normalization_constraints
from skillmodels.constraints import _get_not_measured_constraints
from skillmodels.constraints import _get_stage_constraints
from skillmodels.constraints import _get_transition_constraints
from skillmodels.constraints import add_bounds


def test_add_bounds():
    ind_tups = [("shock_sds", i) for i in range(5)] + [
        ("meas_sds", 4),
        ("bla", "blubb"),
        ("meas_sds", "foo"),
        ("initial_cholcovs", "a-b-c"),
        ("initial_cholcovs", "cog-cog"),
    ]
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(ind_tups, names=["category", "name2"])
    )
    expected = df.copy(deep=True)
    expected["lower_bound"] = [0.1] * 5 + [0.1, -np.inf, 0.1, -np.inf, 0.1]

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
        {
            "loc": [
                ("loadings", 0, "m1", "fac1"),
                ("loadings", 0, "m2", "fac1"),
                ("controls", 0, "m1", "constant"),
                ("loadings", 1, "m1", "fac1"),
                ("loadings", 0, "m3", "fac2"),
            ],
            "type": "fixed",
            "value": [2, 1.5, 0.5, 3, 1],
        }
    ]

    calculated = _get_normalization_constraints(norm)
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
    labels = {"latent_factors": columns}

    expected = [
        {
            "loc": [
                ("loadings", 0, "m1", "fac2"),
                ("loadings", 0, "m3", "fac1"),
                ("loadings", 1, "m1", "fac2"),
                ("loadings", 1, "m3", "fac1"),
            ],
            "type": "fixed",
            "value": 0.0,
        }
    ]

    calculated = _get_not_measured_constraints(uinfo, labels)
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints for mixture weights
# ======================================================================================


def test_mixture_weight_constraints_mixture():
    calculated = _get_mixture_weights_constraints(n_mixtures=2)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weights", "type": "probability"}]
    assert_list_equal_except_for_order(calculated, expected)


def test_mixture_weight_constraints_normal():
    calculated = _get_mixture_weights_constraints(n_mixtures=1)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weights", "type": "fixed", "value": 1.0}]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints for development stages
# ======================================================================================


def test_stage_constraints():
    stages = [0]
    stagemap = [0] * 3

    expected = [
        {
            "locs": [("transition", 0), ("transition", 1), ("transition", 2)],
            "type": "pairwise_equality",
        },
        {
            "locs": [("shock_sds", 0), ("shock_sds", 1), ("shock_sds", 2)],
            "type": "pairwise_equality",
        },
    ]

    calculated = _get_stage_constraints(stagemap, stages)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints for constant factors
# ======================================================================================


def test_constant_factor_constraints():
    labels = {
        "latent_factors": ["fac1", "fac2"],
        "periods": [0, 1, 2],
        "transition_names": ["bla", "constant"],
    }

    expected = [
        {"loc": ("shock_sds", 0, "fac2", "-"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 1, "fac2", "-"), "type": "fixed", "value": 0.0},
    ]

    calculated = _get_constant_factors_constraints(labels)
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
        ("initial_states", 0, "mixture_0", "fac1"),
        ("initial_states", 0, "mixture_1", "fac1"),
        ("initial_states", 0, "mixture_2", "fac1"),
    ]

    expected = [{"loc": ind_tups, "type": "increasing"}]

    calculated = _get_initial_states_constraints(nmixtures, factors)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


# ======================================================================================
# constraints on transition parameters
# ======================================================================================


def test_trans_coeff_constraints():
    labels = {
        "latent_factors": ["fac1", "fac2", "fac3"],
        "transition_names": ["log_ces", "bla", "blubb"],
        "periods": [0, 1, 2],
    }
    labels["all_factors"] = labels["latent_factors"]

    expected = [
        {
            "loc": [
                ("transition", 0, "fac1", "fac1"),
                ("transition", 0, "fac1", "fac2"),
                ("transition", 0, "fac1", "fac3"),
            ],
            "type": "probability",
        },
        {
            "loc": [
                ("transition", 1, "fac1", "fac1"),
                ("transition", 1, "fac1", "fac2"),
                ("transition", 1, "fac1", "fac3"),
            ],
            "type": "probability",
        },
    ]
    calculated = _get_transition_constraints(labels)

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
        "outcomes": {"f1": "outcome", "f2": "outcome"},
        "free_controls": True,
        "free_constant": True,
        "free_loadings": True,
    }
    return anch_info


def test_anchoring_constraints_no_constraint_needed(anch_uinfo, base_anchoring_info):
    calculated = _get_anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))
    assert calculated == []


def test_anchoring_constraints_for_constants(anch_uinfo, base_anchoring_info):
    base_anchoring_info["free_constant"] = False
    calculated = _get_anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))

    del calculated[0]["description"]
    expected = [
        {
            "loc": [
                ("controls", 0, "outcome_f1", "constant"),
                ("controls", 0, "outcome_f2", "constant"),
                ("controls", 1, "outcome_f1", "constant"),
                ("controls", 1, "outcome_f2", "constant"),
            ],
            "type": "fixed",
            "value": 0,
        }
    ]

    assert calculated == expected


def test_anchoring_constraints_for_controls(anch_uinfo, base_anchoring_info):
    base_anchoring_info["free_controls"] = False
    calculated = _get_anchoring_constraints(
        anch_uinfo, ["c1", "c2"], base_anchoring_info, (0, 1)
    )

    for constr in calculated:
        del constr["description"]

    expected = [
        {
            "loc": [
                ("controls", 0, "outcome_f1", "c1"),
                ("controls", 0, "outcome_f1", "c2"),
                ("controls", 0, "outcome_f2", "c1"),
                ("controls", 0, "outcome_f2", "c2"),
                ("controls", 1, "outcome_f1", "c1"),
                ("controls", 1, "outcome_f1", "c2"),
                ("controls", 1, "outcome_f2", "c1"),
                ("controls", 1, "outcome_f2", "c2"),
            ],
            "type": "fixed",
            "value": 0,
        }
    ]

    assert calculated == expected


def test_anchoring_constraints_for_loadings(anch_uinfo, base_anchoring_info):
    base_anchoring_info["free_loadings"] = False
    calculated = _get_anchoring_constraints(anch_uinfo, [], base_anchoring_info, (0, 1))

    expected = [
        {
            "loc": [
                ("loadings", 0, "outcome_f1", "f1"),
                ("loadings", 0, "outcome_f2", "f2"),
                ("loadings", 1, "outcome_f1", "f1"),
                ("loadings", 1, "outcome_f2", "f2"),
            ],
            "type": "fixed",
            "value": 1,
        }
    ]

    for constr in calculated:
        del constr["description"]

    assert calculated == expected


# ======================================================================================


def assert_list_equal_except_for_order(list1, list2):
    for item in list1:
        assert item in list2, f"{item} is in list1 but not in list2"
    for item in list2:
        assert item in list1, f"{item} is in list2 but not in list1"
