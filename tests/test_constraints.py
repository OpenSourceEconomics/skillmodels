"""Tests for constraints."""

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.constraints import (
    _get_anchoring_constraints,
    _get_constant_factors_constraints,
    _get_constraints_for_augmented_periods,
    _get_initial_states_constraints,
    _get_mixture_weights_constraints,
    _get_normalization_constraints,
    _get_stage_constraints,
    _get_transition_constraints,
    add_bounds,
    constraints_dicts_to_om,
    get_constraints_dicts,
)
from skillmodels.process_model import process_model
from skillmodels.test_data.simplest_augmented_model import SIMPLEST_AUGMENTED_MODEL
from skillmodels.types import Anchoring, Labels, Normalizations


def test_add_bounds() -> None:
    ind_tups = [("shock_sds", i) for i in range(5)] + [
        ("meas_sds", 4),
        ("bla", "blubb"),
        ("meas_sds", "foo"),
        ("initial_cholcovs", "a-b-c"),
        ("initial_cholcovs", "cog-cog"),
    ]
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(ind_tups, names=["category", "name2"]),
    )
    expected = df.copy(deep=True)
    expected["lower_bound"] = [0.1] * 5 + [0.1, -np.inf, 0.1, -np.inf, 0.1]
    expected["upper_bound"] = np.inf

    calculated = add_bounds(df, 0.1)
    assert_frame_equal(calculated, expected)


def test_normalization_constraints() -> None:
    norm = {
        "fac1": Normalizations(
            loadings=({"m1": 2, "m2": 1.5}, {"m1": 3}),
            intercepts=({"m1": 0.5}, {}),
        ),
        "fac2": Normalizations(
            loadings=({"m3": 1}, {}),
            intercepts=({}, {}),
        ),
    }

    expected = [
        {
            "loc": ("loadings", 0, "m1", "fac1"),
            "type": "fixed",
            "value": 2,
        },
        {
            "loc": ("loadings", 0, "m2", "fac1"),
            "type": "fixed",
            "value": 1.5,
        },
        {
            "loc": ("controls", 0, "m1", "constant"),
            "type": "fixed",
            "value": 0.5,
        },
        {
            "loc": ("loadings", 1, "m1", "fac1"),
            "type": "fixed",
            "value": 3,
        },
        {
            "loc": ("loadings", 0, "m3", "fac2"),
            "type": "fixed",
            "value": 1,
        },
    ]

    calculated = _get_normalization_constraints(norm, factors=("fac1", "fac2"))
    for c in calculated:
        del c["description"]

    assert_list_equal_except_for_order(calculated, expected)


def test_mixture_weight_constraints_mixture() -> None:
    calculated = _get_mixture_weights_constraints(n_mixtures=2)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weights", "type": "probability"}]
    assert_list_equal_except_for_order(calculated, expected)


def test_mixture_weight_constraints_normal() -> None:
    calculated = _get_mixture_weights_constraints(n_mixtures=1)
    for c in calculated:
        del c["description"]
    expected = [{"loc": "mixture_weights", "type": "fixed", "value": 1.0}]
    assert_list_equal_except_for_order(calculated, expected)


def test_stage_constraints() -> None:
    stages = (0,)
    stagemap = (0, 0, 0)

    expected = [
        {
            "loc": [("transition", 0), ("transition", 1), ("transition", 2)],
            "type": "pairwise_equality",
        },
        {
            "loc": [("shock_sds", 0), ("shock_sds", 1), ("shock_sds", 2)],
            "type": "pairwise_equality",
        },
    ]

    calculated = _get_stage_constraints(stagemap=stagemap, stages=stages)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_stage_constraints_with_endogenous_factors() -> None:
    stages = (0, 1, 2, 3)
    stagemap = (0, 1, 0, 1, 2, 3)
    expected = [
        {
            "loc": [("transition", 0), ("transition", 2)],
            "type": "pairwise_equality",
        },
        {
            "loc": [("transition", 1), ("transition", 3)],
            "type": "pairwise_equality",
        },
        {
            "loc": [("shock_sds", 0), ("shock_sds", 2)],
            "type": "pairwise_equality",
        },
        {
            "loc": [("shock_sds", 1), ("shock_sds", 3)],
            "type": "pairwise_equality",
        },
    ]

    calculated = _get_stage_constraints(stagemap=stagemap, stages=stages)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_constant_factor_constraints() -> None:
    labels = Labels(
        latent_factors=("fac1", "fac2"),
        observed_factors=(),
        controls=("constant",),
        periods=(0, 1, 2),
        stagemap=(0, 0, 0),
        stages=(0,),
        aug_periods=(0, 1, 2),
        aug_periods_to_periods=MappingProxyType({0: 0, 1: 1, 2: 2}),
        aug_stagemap=(0, 0, 0),
        aug_stages=(0,),
        aug_stages_to_stages=MappingProxyType({0: 0}),
        transition_names=("bla", "constant"),
    )

    expected = [
        {"loc": ("shock_sds", 0, "fac2", "-"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 1, "fac2", "-"), "type": "fixed", "value": 0.0},
    ]

    calculated = _get_constant_factors_constraints(labels)
    for c in calculated:
        del c["description"]
    assert_list_equal_except_for_order(calculated, expected)


def test_initial_mean_constraints() -> None:
    nmixtures = 3
    factors = ("fac1", "fac2", "fac3")
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


def test_trans_coeff_constraints() -> None:
    labels = Labels(
        latent_factors=("fac1", "fac2", "fac3"),
        observed_factors=(),
        controls=("constant",),
        periods=(0, 1, 2),
        stagemap=(0, 0, 0),
        stages=(0,),
        aug_periods=(0, 1, 2),
        aug_periods_to_periods=MappingProxyType({0: 0, 1: 1, 2: 2}),
        aug_stagemap=(0, 0, 0),
        aug_stages=(0,),
        aug_stages_to_stages=MappingProxyType({0: 0}),
        transition_names=("log_ces", "bla", "blubb"),
    )

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
    return Anchoring(
        anchoring=True,
        factors=("f1", "f2"),
        outcomes=MappingProxyType({"f1": "outcome", "f2": "outcome"}),
        free_controls=True,
        free_constant=True,
        free_loadings=True,
        ignore_constant_when_anchoring=False,
    )


def test_anchoring_constraints_no_constraint_needed(
    anch_uinfo, base_anchoring_info
) -> None:
    calculated = _get_anchoring_constraints(anch_uinfo, (), base_anchoring_info, (0, 1))
    assert calculated == []


def test_anchoring_constraints_for_constants(anch_uinfo) -> None:
    anchoring_info = Anchoring(
        anchoring=True,
        factors=("f1", "f2"),
        outcomes=MappingProxyType({"f1": "outcome", "f2": "outcome"}),
        free_controls=True,
        free_constant=False,
        free_loadings=True,
        ignore_constant_when_anchoring=False,
    )
    calculated = _get_anchoring_constraints(anch_uinfo, (), anchoring_info, (0, 1))

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
        },
    ]

    assert calculated == expected


def test_anchoring_constraints_for_controls(anch_uinfo) -> None:
    anchoring_info = Anchoring(
        anchoring=True,
        factors=("f1", "f2"),
        outcomes=MappingProxyType({"f1": "outcome", "f2": "outcome"}),
        free_controls=False,
        free_constant=True,
        free_loadings=True,
        ignore_constant_when_anchoring=False,
    )
    calculated = _get_anchoring_constraints(
        anch_uinfo,
        ("c1", "c2"),
        anchoring_info,
        (0, 1),
    )

    for c_t in calculated:
        del c_t["description"]

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
        },
    ]

    assert calculated == expected


def test_anchoring_constraints_for_loadings(anch_uinfo) -> None:
    anchoring_info = Anchoring(
        anchoring=True,
        factors=("f1", "f2"),
        outcomes=MappingProxyType({"f1": "outcome", "f2": "outcome"}),
        free_controls=True,
        free_constant=True,
        free_loadings=False,
        ignore_constant_when_anchoring=False,
    )
    calculated = _get_anchoring_constraints(anch_uinfo, (), anchoring_info, (0, 1))

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
        },
    ]

    for c_t in calculated:
        del c_t["description"]

    assert calculated == expected


def assert_list_equal_except_for_order(list1, list2) -> None:
    for item in list1:
        assert item in list2, f"{item} is in list1 but not in list2"
    for item in list2:
        assert item in list1, f"{item} is in list2 but not in list1"


@pytest.fixture
def simplest_augmented_model():
    return process_model(SIMPLEST_AUGMENTED_MODEL)


def test_get_constraints_dicts_with_endogenous_factors(
    simplest_augmented_model,
) -> None:
    constraints = get_constraints_dicts(
        update_info=simplest_augmented_model.update_info,
        labels=simplest_augmented_model.labels,
        dimensions=simplest_augmented_model.dimensions,
        anchoring_info=simplest_augmented_model.anchoring,
        normalizations=simplest_augmented_model.normalizations,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
    )
    # Should contain augmented-period constraints
    assert any(c.get("value") == 1e-08 for c in constraints)


def test_constraints_dicts_to_om_type_dispatch(simplest_augmented_model) -> None:
    constraints = get_constraints_dicts(
        update_info=simplest_augmented_model.update_info,
        labels=simplest_augmented_model.labels,
        dimensions=simplest_augmented_model.dimensions,
        anchoring_info=simplest_augmented_model.anchoring,
        normalizations=simplest_augmented_model.normalizations,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
    )
    om_constraints = constraints_dicts_to_om(constraints)
    assert len(om_constraints) > 0


def test_constraints_dicts_to_om_equality_type() -> None:
    dicts = [
        {
            "loc": ("loadings", 0, "y1", "f1"),
            "type": "equality",
            "id": 0,
            "description": "test",
        },
    ]
    result = constraints_dicts_to_om(dicts)
    assert len(result) == 1


def test_constraints_dicts_to_om_probability_type() -> None:
    dicts = [
        {
            "loc": "mixture_weights",
            "type": "probability",
            "id": 0,
            "description": "test",
        },
    ]
    result = constraints_dicts_to_om(dicts)
    assert len(result) == 1


def test_constraints_dicts_to_om_increasing_type() -> None:
    dicts = [
        {
            "loc": [
                ("initial_states", 0, "m0", "f1"),
                ("initial_states", 0, "m1", "f1"),
            ],
            "type": "increasing",
            "id": 0,
            "description": "test",
        },
    ]
    result = constraints_dicts_to_om(dicts)
    assert len(result) == 1


def test_constraints_dicts_to_om_unknown_type_raises() -> None:
    dicts = [
        {
            "loc": ("loadings", 0, "y1", "f1"),
            "type": "unknown_type",
            "id": 0,
            "description": "test",
        },
    ]
    with pytest.raises(TypeError, match="unknown_type"):
        constraints_dicts_to_om(dicts)


def test_get_constraints_for_augmented_periods(simplest_augmented_model) -> None:
    calculated = _get_constraints_for_augmented_periods(
        labels=simplest_augmented_model.labels,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
    )
    for c in calculated:
        del c["description"]
    expected = [
        {"loc": ("transition", 0, "fac1", "fac1"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 0, "fac1", "fac2"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 0, "fac1", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 0, "fac1", "constant"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 0, "fac1", "-"), "type": "fixed", "value": 0.00000001},
        {"loc": ("transition", 2, "fac1", "fac1"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 2, "fac1", "fac2"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 2, "fac1", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 2, "fac1", "constant"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 1, "fac2", "fac1"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 1, "fac2", "fac2"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 1, "fac2", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 1, "fac2", "constant"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 1, "fac2", "-"), "type": "fixed", "value": 0.00000001},
        {"loc": ("transition", 3, "fac2", "fac1"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 3, "fac2", "fac2"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 3, "fac2", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 3, "fac2", "constant"), "type": "fixed", "value": 0.0},
    ]
    assert_list_equal_except_for_order(calculated, expected)
