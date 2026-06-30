"""Tests for check_model validation functions."""

from types import SimpleNamespace

import pytest

from skillmodels.common.check_model import (
    _check_anchoring,
    _check_loadings_are_not_normalized_to_zero,
    _check_normalized_variables_are_present,
    check_stagemap,
)
from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.process_model import process_model
from skillmodels.test_data.simplest_augmented_model import SIMPLEST_AUGMENTED_MODEL


def test_check_model_rejects_two_endogenous_factors_sharing_measurement() -> None:
    # A second endogenous factor reusing fac2's "inv" measurement — the kind of
    # duplicate-measurement collision the old guard silently missed.
    base = SIMPLEST_AUGMENTED_MODEL
    inv_b = FactorSpec(
        measurements=(("inv",), ("inv",)),
        normalizations=Normalizations(
            loadings=({"inv": 1}, {"inv": 1}), intercepts=({}, {})
        ),
        is_endogenous=True,
        transition_function="linear",
    )
    model = base._replace(factors=dict(base.factors) | {"fac2b": inv_b})
    with pytest.raises(ValueError, match="overlap"):
        process_model(model)


def test_invalid_stagemap_length() -> None:
    result = check_stagemap(
        stagemap=(0, 0),
        stages=(0,),
        n_periods=5,
        is_augmented=False,
    )
    assert any("length" in msg.lower() or "n_periods" in msg.lower() for msg in result)


def test_invalid_anchoring_non_bool() -> None:
    anchoring = SimpleNamespace(
        anchoring="yes",  # not bool
        outcomes={},
        free_controls=False,
        free_constant=False,
        free_loadings=False,
    )
    result = _check_anchoring(anchoring)
    assert any("bool" in msg for msg in result)


def test_invalid_anchoring_non_mapping_outcomes() -> None:
    anchoring = SimpleNamespace(
        anchoring=True,
        outcomes="not_a_mapping",
        free_controls=False,
        free_constant=False,
        free_loadings=False,
    )
    result = _check_anchoring(anchoring)
    assert any("Mapping" in msg for msg in result)


def test_invalid_anchoring_outcome_type() -> None:
    anchoring = SimpleNamespace(
        anchoring=True,
        outcomes={"f1": [1, 2, 3]},  # list is not str/int/tuple
        free_controls=False,
        free_constant=False,
        free_loadings=False,
    )
    result = _check_anchoring(anchoring)
    assert any("variable" in msg.lower() for msg in result)


def test_invalid_anchoring_free_controls_type() -> None:
    anchoring = SimpleNamespace(
        anchoring=True,
        outcomes={},
        free_controls="yes",  # not bool
        free_constant=False,
        free_loadings=False,
    )
    result = _check_anchoring(anchoring)
    assert any("free_controls" in msg for msg in result)


def test_invalid_measurements_not_tuples() -> None:
    """Bad measurements shape is caught at the FactorSpec beartype perimeter.

    Pre-beartype, the spec built and the model-check aggregator
    surfaced a soft error message. Now the construction itself
    raises `ModelSpecInitializationError`. The soft-check arm of
    `_check_measurements` is dead code (kept only for non-type
    issues that beartype can't see).
    """
    from skillmodels.exceptions import ModelSpecInitializationError  # noqa: PLC0415

    with pytest.raises(ModelSpecInitializationError, match="measurements"):
        FactorSpec(measurements=(["y1", "y2"],))  # ty: ignore[invalid-argument-type]


def test_invalid_measurement_type() -> None:
    """Bad measurement element type fails at `FactorSpec.__init__` (beartype)."""
    from skillmodels.exceptions import ModelSpecInitializationError  # noqa: PLC0415

    with pytest.raises(ModelSpecInitializationError, match="measurements"):
        FactorSpec(measurements=((["nested_list"],),))  # ty: ignore[invalid-argument-type]


def test_normalized_variable_not_in_measurements() -> None:
    spec = ModelSpec(
        factors={
            "f1": FactorSpec(
                measurements=(("y1", "y2"),),
                normalizations=Normalizations(
                    loadings=({"y99": 1},),
                    intercepts=({},),
                ),
            ),
        },
    )
    result = _check_normalized_variables_are_present(
        list_of_normdicts=[{"y99": 1}],
        model_spec=spec,
        factor="f1",
    )
    assert any("y99" in msg for msg in result)


def test_invalid_anchoring_free_constant_type() -> None:
    anchoring = SimpleNamespace(
        anchoring=True,
        outcomes={},
        free_controls=False,
        free_constant="yes",
        free_loadings=False,
    )
    result = _check_anchoring(anchoring)
    assert any("free_constant" in msg for msg in result)


def test_invalid_anchoring_free_loadings_type() -> None:
    anchoring = SimpleNamespace(
        anchoring=True,
        outcomes={},
        free_controls=False,
        free_constant=False,
        free_loadings="yes",
    )
    result = _check_anchoring(anchoring)
    assert any("free_loadings" in msg for msg in result)


def test_loading_normalized_to_zero() -> None:
    result = _check_loadings_are_not_normalized_to_zero(
        list_of_normdicts=[{"y1": 0}],
        factor="f1",
    )
    assert any("0" in msg for msg in result)
