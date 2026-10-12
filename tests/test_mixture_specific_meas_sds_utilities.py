"""Keep mixture-SD declarations aligned when explicitly reducing a model."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.common.model_spec import FactorSpec, ModelSpec, Normalizations
from skillmodels.common.utilities import (
    _get_params_index,
    extract_factors,
    reduce_n_periods,
    remove_factors,
    remove_measurements,
)

A_MEASURES = (("a0", "a_keep", "shared"), ("a0", "a_keep", "shared", "late"))
B_MEASURES = (("b0", "b_only", "shared"),) * 2
SPECIFIC = ("b_only", "a_keep", "late", "shared")
OPERATIONS = ("measurements", "factors", "extract", "periods")


def _factor(measurements: tuple[tuple[str, ...], ...], anchor: str) -> FactorSpec:
    return FactorSpec(
        measurements=measurements,
        normalizations=Normalizations(
            loadings=({anchor: 1.0},) * len(measurements),
            intercepts=({anchor: 0.0},) * len(measurements),
        ),
        transition_function="linear",
    )


@pytest.fixture(params=(2, 11))
def model(request: pytest.FixtureRequest) -> ModelSpec:
    return ModelSpec(
        factors={"a": _factor(A_MEASURES, "a0"), "b": _factor(B_MEASURES, "b0")},
        n_mixtures=request.param,
        mixture_specific_meas_sds=SPECIFIC,
    )


def _reduce(
    operation: str,
    model: ModelSpec,
    params: pd.DataFrame | None,
) -> ModelSpec | tuple[ModelSpec, pd.DataFrame]:
    if operation == "measurements":
        return remove_measurements("b_only", model, params)
    if operation == "factors":
        return remove_factors("b", model, params)
    if operation == "extract":
        return extract_factors("a", model, params)
    if operation == "periods":
        return reduce_n_periods(model, 1, params)
    raise AssertionError(operation)


def _expected_model(operation: str, model: ModelSpec, option_on: bool) -> ModelSpec:
    """Construct the intended reduced model directly from named cases."""
    if operation == "measurements":
        factors = {
            "a": _factor(A_MEASURES, "a0"),
            "b": _factor((("b0", "shared"),) * 2, "b0"),
        }
        selected = ("a_keep", "late", "shared")
    elif operation in ("factors", "extract"):
        factors = {"a": _factor(A_MEASURES, "a0")}
        selected = ("a_keep", "late", "shared")
    elif operation == "periods":
        factors = {
            "a": _factor((A_MEASURES[0],), "a0"),
            "b": _factor((B_MEASURES[0],), "b0"),
        }
        selected = ("b_only", "a_keep", "shared")
    else:
        raise AssertionError(operation)
    return ModelSpec(
        factors=factors,
        n_mixtures=model.n_mixtures,
        mixture_specific_meas_sds=selected if option_on else (),
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("with_params", (False, True))
@pytest.mark.parametrize("option_on", (False, True))
def test_reductions_match_direct_model_and_preserve_params(
    model: ModelSpec,
    operation: str,
    with_params: bool,
    option_on: bool,
) -> None:
    original = model if option_on else model._replace(mixture_specific_meas_sds=())
    index = _get_params_index(original)
    original_params = pd.DataFrame(
        {"value": np.arange(len(index), dtype=float)},
        index=index,
    )
    before = original_params.copy(deep=True)
    result = _reduce(operation, original, original_params if with_params else None)
    expected = _expected_model(operation, original, option_on)
    if with_params:
        assert isinstance(result, tuple)
        reduced, reduced_params = result
        assert isinstance(reduced_params, pd.DataFrame)
        assert_frame_equal(reduced_params, before.loc[_get_params_index(expected)])
    else:
        assert isinstance(result, ModelSpec)
        reduced = result
    assert reduced == expected
    assert_frame_equal(original_params, before)
    assert original.factors["a"].measurements == A_MEASURES
    assert original.factors["b"].measurements == B_MEASURES
    assert original.mixture_specific_meas_sds == (SPECIFIC if option_on else ())


def test_removing_all_selected_measures_returns_to_pooled_option(
    model: ModelSpec,
) -> None:
    reduced = remove_measurements(list(SPECIFIC), model)
    assert isinstance(reduced, ModelSpec)
    assert reduced.mixture_specific_meas_sds == ()
    assert reduced.factors["a"].measurements == (("a0",),) * 2
    assert reduced.factors["b"].measurements == (("b0",),) * 2
    assert model.mixture_specific_meas_sds == SPECIFIC


def test_nonreducing_copy_preserves_selection(model: ModelSpec) -> None:
    assert model.with_controls(("age",)).mixture_specific_meas_sds == SPECIFIC


def test_constructor_and_explicit_replacement_still_reject_unknown_names(
    model: ModelSpec,
) -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        ModelSpec(
            factors=model.factors,
            n_mixtures=model.n_mixtures,
            mixture_specific_meas_sds=("misspelled",),
        )
    with pytest.raises(ValueError, match="not a measurement"):
        model._replace(factors={"a": model.factors["a"]})
