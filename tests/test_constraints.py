"""Tests for constraints."""

from types import MappingProxyType
from typing import Any

import numpy as np
import optimagic as om
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.common.constraints import (
    FixedConstraintWithValue,
    _get_anchoring_constraints,
    _get_constant_factors_constraints,
    _get_constraints_for_augmented_periods,
    _get_initial_states_constraints,
    _get_mixture_weights_constraints,
    _get_normalization_constraints,
    _get_stage_constraints,
    _get_transition_constraints,
    add_bounds,
    get_constraints,
)
from skillmodels.common.process_model import process_model
from skillmodels.common.types import (
    Anchoring,
    Labels,
    MeasurementType,
    Normalizations,
)
from skillmodels.test_data.simplest_augmented_model import SIMPLEST_AUGMENTED_MODEL


def _corr_model_processed():
    """Process a correction model: an endogenous investment + an instrument."""
    from dataclasses import replace  # noqa: PLC0415

    from skillmodels.common.model_spec import CorrectionSpec  # noqa: PLC0415
    from skillmodels.test_data.model2 import MODEL2  # noqa: PLC0415

    fac3 = MODEL2.factors["fac3"]
    corr = CorrectionSpec(instruments=("inv_z",))
    new_factors = dict(MODEL2.factors) | {
        "fac3": replace(fac3, is_endogenous=True, correction=corr)
    }
    model = MODEL2._replace(factors=new_factors)._replace(stagemap=None)
    model = model._replace(observed_factors=("inv_z",))
    return process_model(model)


def test_get_constraints_pins_kappa_to_zero_on_carry_forward_periods() -> None:
    processed = _corr_model_processed()
    constraints = get_constraints(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        anchoring_info=processed.anchoring,
        normalizations=processed.normalizations,
        endogenous_factors_info=processed.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    kappa_fixed = [
        c
        for c in constraints
        if isinstance(c, FixedConstraintWithValue)
        and isinstance(c.loc, tuple)
        and c.loc[0] == "kappa"
    ]
    assert kappa_fixed, "kappa must be pinned to 0 on carry-forward periods"
    # All such constraints pin kappa to exactly 0.
    assert all(c.value == 0.0 for c in kappa_fixed)
    # They fall only on the state factors' carry-forward (STATES) aug periods,
    # never on the production (ENDOGENOUS) aug periods where kappa is free.
    meas_types = processed.endogenous_factors_info.aug_periods_to_aug_period_meas_types
    for c in kappa_fixed:
        assert isinstance(c.loc, tuple)
        _category, aug_period, target, _term = c.loc
        assert target in ("fac1", "fac2")
        assert meas_types[aug_period] == MeasurementType.STATES


def test_get_constraints_pins_instrument_out_of_production() -> None:
    # Built-in production transitions enumerate a free coefficient for every
    # observed factor, including the excluded instrument; that coefficient must be
    # pinned to 0 on the production (ENDOGENOUS) aug periods so the instrument
    # cannot leak into production.
    processed = _corr_model_processed()
    constraints = get_constraints(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        anchoring_info=processed.anchoring,
        normalizations=processed.normalizations,
        endogenous_factors_info=processed.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    meas_types = processed.endogenous_factors_info.aug_periods_to_aug_period_meas_types
    inst_pins = [
        c
        for c in constraints
        if isinstance(c, FixedConstraintWithValue)
        and isinstance(c.loc, tuple)
        and c.loc[0] == "transition"
        and "inv_z" in c.loc[3]
        and meas_types[c.loc[1]] == MeasurementType.ENDOGENOUS_FACTORS
    ]
    assert inst_pins, "instrument coeffs must be pinned to 0 on production periods"
    assert all(c.value == 0.0 for c in inst_pins)
    for c in inst_pins:
        assert isinstance(c.loc, tuple)
        _category, _aug_period, target, _name2 = c.loc
        assert target in ("fac1", "fac2")


def test_get_constraints_skips_custom_target_transition() -> None:
    # A custom (registered) production transition on a correction target has no
    # built-in `params_<name>` enumerator; the instrument-exclusion guard must
    # skip it (custom-production leakage is validated separately by
    # `check_model`) rather than raise AttributeError.
    from dataclasses import replace  # noqa: PLC0415

    from skillmodels.common.decorators import register_params  # noqa: PLC0415
    from skillmodels.common.model_spec import CorrectionSpec  # noqa: PLC0415
    from skillmodels.test_data.model2 import MODEL2  # noqa: PLC0415

    @register_params(params=["constant", "fac1", "fac2"])
    def custom_prod(fac1, fac2, params):
        return params["constant"] + params["fac1"] * fac1 + params["fac2"] * fac2

    corr = CorrectionSpec(instruments=("inv_z",))
    new_factors = dict(MODEL2.factors) | {
        "fac1": replace(MODEL2.factors["fac1"], transition_function=custom_prod),
        "fac3": replace(MODEL2.factors["fac3"], is_endogenous=True, correction=corr),
    }
    model = (
        MODEL2._replace(factors=new_factors)
        ._replace(stagemap=None)
        ._replace(observed_factors=("inv_z",))
    )
    processed = process_model(model)

    # Must not raise `AttributeError: ... has no attribute 'params_custom_prod'`.
    constraints = get_constraints(
        update_info=processed.update_info,
        labels=processed.labels,
        dimensions=processed.dimensions,
        anchoring_info=processed.anchoring,
        normalizations=processed.normalizations,
        endogenous_factors_info=processed.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    assert constraints


def _to_dict(c: om.constraints.Constraint) -> dict[str, Any]:
    """Convert a constraint object to a comparable dict for testing."""
    if isinstance(c, FixedConstraintWithValue):
        return {"loc": c.loc, "type": "fixed", "value": c.value}
    if isinstance(c, om.PairwiseEqualityConstraint):
        locs = [s.keywords["loc"] for s in c.selectors]  # ty: ignore[unresolved-attribute]
        return {"loc": locs, "type": "pairwise_equality"}
    if isinstance(c, om.ProbabilityConstraint):
        return {"loc": c.selector.keywords["loc"], "type": "probability"}  # ty: ignore[unresolved-attribute]
    if isinstance(c, om.IncreasingConstraint):
        return {"loc": c.selector.keywords["loc"], "type": "increasing"}  # ty: ignore[unresolved-attribute]
    if isinstance(c, om.EqualityConstraint):
        return {"loc": c.selector.keywords["loc"], "type": "equality"}  # ty: ignore[unresolved-attribute]
    raise TypeError(type(c))


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


def test_mixture_weight_constraints_mixture() -> None:
    calculated = _get_mixture_weights_constraints(n_mixtures=2)
    as_dicts = [_to_dict(c) for c in calculated]
    expected = [{"loc": "mixture_weights", "type": "probability"}]
    assert_list_equal_except_for_order(as_dicts, expected)


def test_mixture_weight_constraints_normal() -> None:
    calculated = _get_mixture_weights_constraints(n_mixtures=1)
    as_dicts = [_to_dict(c) for c in calculated]
    expected = [{"loc": "mixture_weights", "type": "fixed", "value": 1.0}]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]
    assert_list_equal_except_for_order(as_dicts, expected)


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
    as_dicts = [_to_dict(c) for c in calculated]

    expected = [
        {
            "loc": (
                ("controls", 0, "outcome_f1", "constant"),
                ("controls", 0, "outcome_f2", "constant"),
                ("controls", 1, "outcome_f1", "constant"),
                ("controls", 1, "outcome_f2", "constant"),
            ),
            "type": "fixed",
            "value": 0,
        },
    ]

    assert as_dicts == expected


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
    as_dicts = [_to_dict(c) for c in calculated]

    expected = [
        {
            "loc": (
                ("controls", 0, "outcome_f1", "c1"),
                ("controls", 0, "outcome_f1", "c2"),
                ("controls", 0, "outcome_f2", "c1"),
                ("controls", 0, "outcome_f2", "c2"),
                ("controls", 1, "outcome_f1", "c1"),
                ("controls", 1, "outcome_f1", "c2"),
                ("controls", 1, "outcome_f2", "c1"),
                ("controls", 1, "outcome_f2", "c2"),
            ),
            "type": "fixed",
            "value": 0,
        },
    ]

    assert as_dicts == expected


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
    as_dicts = [_to_dict(c) for c in calculated]

    expected = [
        {
            "loc": (
                ("loadings", 0, "outcome_f1", "f1"),
                ("loadings", 0, "outcome_f2", "f2"),
                ("loadings", 1, "outcome_f1", "f1"),
                ("loadings", 1, "outcome_f2", "f2"),
            ),
            "type": "fixed",
            "value": 1,
        },
    ]

    assert as_dicts == expected


def assert_list_equal_except_for_order(list1, list2) -> None:
    for item in list1:
        assert item in list2, f"{item} is in list1 but not in list2"
    for item in list2:
        assert item in list1, f"{item} is in list2 but not in list1"


@pytest.fixture
def simplest_augmented_model():
    return process_model(SIMPLEST_AUGMENTED_MODEL)


def test_get_constraints_with_endogenous_factors(
    simplest_augmented_model,
) -> None:
    constraints = get_constraints(
        update_info=simplest_augmented_model.update_info,
        labels=simplest_augmented_model.labels,
        dimensions=simplest_augmented_model.dimensions,
        anchoring_info=simplest_augmented_model.anchoring,
        normalizations=simplest_augmented_model.normalizations,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    # Should contain augmented-period constraints
    assert any(
        isinstance(c, FixedConstraintWithValue) and c.value == 1e-08
        for c in constraints
    )


def test_get_constraints_returns_om_objects(simplest_augmented_model) -> None:
    constraints = get_constraints(
        update_info=simplest_augmented_model.update_info,
        labels=simplest_augmented_model.labels,
        dimensions=simplest_augmented_model.dimensions,
        anchoring_info=simplest_augmented_model.anchoring,
        normalizations=simplest_augmented_model.normalizations,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    assert len(constraints) > 0
    for c in constraints:
        assert isinstance(c, om.constraints.Constraint)


def test_get_constraints_for_augmented_periods(simplest_augmented_model) -> None:
    calculated = _get_constraints_for_augmented_periods(
        labels=simplest_augmented_model.labels,
        endogenous_factors_info=simplest_augmented_model.endogenous_factors_info,
        bounds_distance=1e-8,
    )
    as_dicts = [_to_dict(c) for c in calculated]
    # Only the non-final aug-period of each meas-type should produce
    # identity constraints: `get_transition_index_tuples` truncates
    # transitions at `aug_periods[:-2]` when endogenous factors are
    # present, so emitting fixed constraints at the last STATES- or
    # ENDO-typed aug-period would target locs that don't exist in the
    # params index. Aug 2 (last STATES-typed) and aug 3 (last
    # ENDO-typed) are therefore intentionally absent from the expected
    # list.
    expected = [
        {"loc": ("transition", 0, "fac1", "fac1"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 0, "fac1", "fac2"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 0, "fac1", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 0, "fac1", "constant"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 0, "fac1", "-"), "type": "fixed", "value": 0.00000001},
        {"loc": ("transition", 1, "fac2", "fac1"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 1, "fac2", "fac2"), "type": "fixed", "value": 1.0},
        {"loc": ("transition", 1, "fac2", "of"), "type": "fixed", "value": 0.0},
        {"loc": ("transition", 1, "fac2", "constant"), "type": "fixed", "value": 0.0},
        {"loc": ("shock_sds", 1, "fac2", "-"), "type": "fixed", "value": 0.00000001},
    ]
    assert_list_equal_except_for_order(as_dicts, expected)
