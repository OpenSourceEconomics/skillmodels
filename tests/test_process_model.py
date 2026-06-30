"""Tests for process model."""

import inspect
from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skillmodels.common.config import TEST_DATA_DIR
from skillmodels.common.control_function import generate_kappa_terms
from skillmodels.common.model_spec import CorrectionSpec, FactorSpec, ModelSpec
from skillmodels.common.process_model import (
    _resolve_control_function,
    get_has_endogenous_factors,
    process_model,
)
from skillmodels.common.types import (
    ControlFunctionInfo,
    Normalizations,
    TransitionInfo,
)
from skillmodels.test_data.model2 import MODEL2


@pytest.fixture
def model2():
    return MODEL2


def test_has_endogenous_factors(model2) -> None:
    assert not process_model(model2).endogenous_factors_info.has_endogenous_factors


def test_dimensions(model2) -> None:
    res = process_model(model2).dimensions
    assert res.n_latent_factors == 3
    assert res.n_observed_factors == 0
    assert res.n_all_factors == 3
    assert res.n_periods == 8
    assert res.n_controls == 2
    assert res.n_mixtures == 1


def test_labels(model2) -> None:
    res = process_model(model2).labels
    assert res.latent_factors == ("fac1", "fac2", "fac3")
    assert res.observed_factors == ()
    assert res.all_factors == ("fac1", "fac2", "fac3")
    assert res.controls == ("constant", "x1")
    assert res.periods == (0, 1, 2, 3, 4, 5, 6, 7)
    assert res.stagemap == (0, 0, 0, 0, 0, 0, 0)
    assert res.stages == (0,)


def test_anchoring(model2) -> None:
    res = process_model(model2).anchoring
    assert res.outcomes == {"fac1": "Q1"}
    assert res.factors == ("fac1",)
    assert res.free_controls
    assert res.free_constant
    assert res.free_loadings


def test_transition_info(model2) -> None:
    res = process_model(model2).transition_info

    assert isinstance(res, TransitionInfo)
    assert callable(res.func)

    assert list(inspect.signature(res.func).parameters) == ["params", "states"]


def test_update_info(model2) -> None:
    res = process_model(model2).update_info
    expected = pd.read_csv(
        TEST_DATA_DIR / "model2_correct_update_info.csv",
        index_col=["aug_period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_normalizations(model2) -> None:
    expected = {
        "fac1": Normalizations(
            loadings=(
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
                {"y1": 1},
            ),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
        "fac2": Normalizations(
            loadings=(
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
                {"y4": 1},
            ),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
        "fac3": Normalizations(
            loadings=({"y7": 1}, {}, {}, {}, {}, {}, {}, {}),
            intercepts=({}, {}, {}, {}, {}, {}, {}, {}),
        ),
    }
    res = process_model(model2).normalizations

    assert res == expected


def _make_fac3_endogenous(model):
    """Return a new model with fac3 set as endogenous."""
    fac3 = model.factors["fac3"]
    new_fac3 = replace(fac3, is_endogenous=True)
    new_factors = dict(model.factors) | {"fac3": new_fac3}
    return model._replace(factors=new_factors)


def test_anchoring_and_endogenous_factors_work_together() -> None:
    model = _make_fac3_endogenous(MODEL2)._replace(stagemap=None)
    # Should not raise - anchoring and endogenous factors now work together
    result = process_model(model)
    # Verify anchoring is enabled
    assert result.anchoring.anchoring
    assert result.anchoring.factors == ("fac1",)
    # Verify endogenous factors are enabled
    assert result.endogenous_factors_info.has_endogenous_factors
    # Verify dimensions
    assert result.dimensions.n_periods == 8
    assert result.dimensions.n_aug_periods == 16
    # Verify update_info has anchoring entries for all aug_periods
    anchoring_updates = result.update_info[result.update_info["purpose"] == "anchoring"]
    assert (
        len(anchoring_updates) == 16
    )  # One per aug_period for the one anchored factor


def test_stagemap_with_endogenous_factors_wrong_labels() -> None:
    model = _make_fac3_endogenous(MODEL2)._replace(
        stagemap=(0, 0, 1, 1, 2, 2, 4),
        anchoring=None,
    )
    with pytest.raises(ValueError, match="Invalid stage map:"):
        process_model(model)


def test_stagemap_with_endogenous_factors() -> None:
    stagemap = (0, 0, 1, 1, 2, 2, 3)
    model = _make_fac3_endogenous(MODEL2)._replace(
        stagemap=stagemap,
        anchoring=None,
    )
    processed = process_model(model)
    assert processed.labels.stagemap == stagemap
    assert processed.labels.stages == (0, 1, 2, 3)
    assert processed.labels.aug_stagemap == (0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7)


@pytest.fixture
def model2_inv():
    return _make_fac3_endogenous(MODEL2)._replace(
        stagemap=None,
        anchoring=None,
    )


def test_with_endog_has_endogenous_factors(model2_inv) -> None:
    assert process_model(model2_inv).endogenous_factors_info.has_endogenous_factors


def test_with_endog_dimensions(model2_inv) -> None:
    res = process_model(model2_inv).dimensions
    assert res.n_latent_factors == 3
    assert res.n_observed_factors == 0
    assert res.n_all_factors == 3
    assert res.n_aug_periods == 16
    assert res.n_periods == 8
    assert res.n_controls == 2
    assert res.n_mixtures == 1


def test_with_endog_labels(model2_inv) -> None:
    res = process_model(model2_inv).labels
    n_aug_periods = 16
    assert res.latent_factors == ("fac1", "fac2", "fac3")
    assert res.observed_factors == ()
    assert res.all_factors == ("fac1", "fac2", "fac3")
    assert res.controls == ("constant", "x1")
    assert res.aug_periods == tuple(range(n_aug_periods))
    assert res.periods == (0, 1, 2, 3, 4, 5, 6, 7)
    assert res.aug_stagemap == tuple(range(n_aug_periods - 2))
    assert res.aug_stages == tuple(range(n_aug_periods - 2))


def test_with_endog_anchoring_is_empty(model2_inv) -> None:
    res = process_model(model2_inv).anchoring
    assert res.outcomes == {}
    assert res.factors == ()
    assert res.free_controls is False
    assert res.free_constant is False
    assert res.free_loadings is False


def test_with_endog_transition_info(model2_inv) -> None:
    res = process_model(model2_inv).transition_info

    assert isinstance(res, TransitionInfo)
    assert callable(res.func)

    assert list(inspect.signature(res.func).parameters) == ["params", "states"]


def test_with_endog_update_info(model2_inv) -> None:
    res = process_model(model2_inv).update_info
    expected = pd.read_csv(
        TEST_DATA_DIR / "model2_with_endog_correct_update_info.csv",
        index_col=["aug_period", "variable"],
    )
    assert_frame_equal(res, expected)


def test_with_endog_normalizations(model2_inv) -> None:
    e = {}
    expected = {
        "fac1": Normalizations(
            loadings=(
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
                {"y1": 1},
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
        "fac2": Normalizations(
            loadings=(
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
                {"y4": 1},
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
        "fac3": Normalizations(
            loadings=(
                e,
                {"y7": 1},
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
                e,
            ),
            intercepts=(e, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e),
        ),
    }
    res = process_model(model2_inv).normalizations

    assert res == expected


def _fspec(**kwargs) -> FactorSpec:
    """Create a minimal FactorSpec for unit tests."""
    return FactorSpec(measurements=((),), **kwargs)


def test_model_has_endogenous_factors_not_specified() -> None:
    factors = {"a": _fspec()}
    assert not get_has_endogenous_factors(factors)


def test_get_has_endogenous_factors_indeed() -> None:
    factors = {
        "a": _fspec(is_endogenous=True),
        "b": _fspec(is_endogenous=False),
    }
    assert get_has_endogenous_factors(factors)


def _corr_model(correction: CorrectionSpec) -> ModelSpec:
    """Two state factors + one endogenous investment factor carrying a correction."""
    factors = {
        "health_mom": _fspec(transition_function="linear"),
        "health_kid": _fspec(transition_function="linear"),
        "ln_inv": _fspec(
            is_endogenous=True,
            transition_function="linear",
            correction=correction,
        ),
    }
    return ModelSpec(
        factors=factors,
        observed_factors=("sum_inv_paid_log", "sum_inv_private_log"),
    )


def test_resolve_control_function_returns_none_without_correction() -> None:
    model = ModelSpec(factors={"a": _fspec(transition_function="linear")})
    assert _resolve_control_function(model) is None


def test_resolve_control_function_resolves_defaults() -> None:
    model = _corr_model(
        CorrectionSpec(instruments=("sum_inv_paid_log", "sum_inv_private_log"))
    )
    info = _resolve_control_function(model)
    assert isinstance(info, ControlFunctionInfo)
    assert info.investment_factor == "ln_inv"
    # Empty predictors/targets default to all state factors.
    assert info.state_predictors == ("health_mom", "health_kid")
    assert info.targets == ("health_mom", "health_kid")
    assert info.instruments == ("sum_inv_paid_log", "sum_inv_private_log")
    # Each target with no explicit kappa_terms defaults to ("cf",).
    assert info.kappa_terms["health_mom"] == ("cf",)
    assert info.kappa_terms["health_kid"] == ("cf",)


def test_resolve_control_function_expands_kappa_degree() -> None:
    model = _corr_model(
        CorrectionSpec(instruments=("sum_inv_paid_log",), kappa_degree=2)
    )
    info = _resolve_control_function(model)
    assert info is not None
    expected = generate_kappa_terms(("health_mom", "health_kid"), max_degree=2)
    assert "cf ** 2" in expected
    assert info.kappa_terms["health_mom"] == expected
    assert info.kappa_terms["health_kid"] == expected


def test_resolve_control_function_preserves_explicit_fields() -> None:
    model = _corr_model(
        CorrectionSpec(
            state_predictors=("health_mom",),
            instruments=("sum_inv_paid_log",),
            targets=("health_kid",),
            kappa_terms={"health_kid": ("cf", "cf ** 2")},
        )
    )
    info = _resolve_control_function(model)
    assert info is not None
    assert info.state_predictors == ("health_mom",)
    assert info.targets == ("health_kid",)
    assert info.kappa_terms["health_kid"] == ("cf", "cf ** 2")
    # Targets not listed get no kappa block.
    assert "health_mom" not in info.kappa_terms


def test_resolve_control_function_rejects_multiple_investment_factors() -> None:
    cf = CorrectionSpec(instruments=("sum_inv_paid_log",))
    factors = {
        "health_mom": _fspec(transition_function="linear"),
        "ln_inv_a": _fspec(
            is_endogenous=True, transition_function="linear", correction=cf
        ),
        "ln_inv_b": _fspec(
            is_endogenous=True, transition_function="linear", correction=cf
        ),
    }
    model = ModelSpec(factors=factors, observed_factors=("sum_inv_paid_log",))
    with pytest.raises(NotImplementedError, match="one investment factor"):
        _resolve_control_function(model)


def test_process_model_wires_control_function_through_augmentation() -> None:
    """The resolved control function must survive endogenous-period augmentation.

    Regression: `_augment_periods_for_endogenous_factors` rebuilds each
    `FactorSpec` and previously omitted `correction`, silently resetting it to
    `None`. Because augmentation runs exactly when endogenous factors exist (the
    only case that can carry a correction), `control_function` was always `None`
    in the real `process_model` pipeline. The resolver unit tests missed it by
    calling `_resolve_control_function` on the un-augmented spec directly.
    """
    fac3 = MODEL2.factors["fac3"]
    corr = CorrectionSpec(instruments=("inv_z",))
    new_fac3 = replace(fac3, is_endogenous=True, correction=corr)
    new_factors = dict(MODEL2.factors) | {"fac3": new_fac3}
    model = MODEL2._replace(factors=new_factors)._replace(stagemap=None)
    # The instrument must be a declared observed factor (the prediction node
    # resolves its position in all_factors).
    model = model._replace(observed_factors=("inv_z",))

    processed = process_model(model)
    cf_info = processed.endogenous_factors_info.control_function
    assert isinstance(cf_info, ControlFunctionInfo)
    assert cf_info.investment_factor == "fac3"
    assert cf_info.instruments == ("inv_z",)
    # The endogenous investment is not a predictor/target of itself; the state
    # factors fac1/fac2 are the defaults.
    assert "fac3" not in cf_info.targets
    assert set(cf_info.targets) == {"fac1", "fac2"}
    assert set(cf_info.state_predictors) == {"fac1", "fac2"}


def test_resolve_control_function_rejects_model_with_no_state_factors() -> None:
    cf = CorrectionSpec(instruments=("z1",))
    factors = {
        "ln_inv": _fspec(
            is_endogenous=True, transition_function="linear", correction=cf
        ),
        "other_inv": _fspec(is_endogenous=True, transition_function="linear"),
    }
    model = ModelSpec(factors=factors, observed_factors=("z1",))
    with pytest.raises(ValueError, match="no state factors"):
        _resolve_control_function(model)


def test_resolve_control_function_rejects_instrument_in_custom_production() -> None:
    from skillmodels.common.decorators import register_params  # noqa: PLC0415

    @register_params(params=["constant", "health_mom", "sum_inv_paid_log"])
    def f_leaky(health_mom: object, sum_inv_paid_log: object, params: dict) -> object:
        return (
            params["constant"]
            + params["health_mom"] * health_mom
            + params["sum_inv_paid_log"] * sum_inv_paid_log
        )

    factors = {
        "health_mom": _fspec(transition_function=f_leaky),
        "health_kid": _fspec(transition_function="linear"),
        "ln_inv": _fspec(
            is_endogenous=True,
            transition_function="linear",
            correction=CorrectionSpec(instruments=("sum_inv_paid_log",)),
        ),
    }
    model = ModelSpec(factors=factors, observed_factors=("sum_inv_paid_log",))
    with pytest.raises(ValueError, match="instrument"):
        _resolve_control_function(model)


def test_resolve_control_function_rejects_instrument_not_observed() -> None:
    factors = {
        "health_mom": _fspec(transition_function="linear"),
        "ln_inv": _fspec(
            is_endogenous=True,
            transition_function="linear",
            correction=CorrectionSpec(instruments=("not_observed",)),
        ),
    }
    model = ModelSpec(factors=factors, observed_factors=("sum_inv_paid_log",))
    with pytest.raises(ValueError, match="observed"):
        _resolve_control_function(model)


def test_resolve_control_function_rejects_correction_on_non_endogenous() -> None:
    factors = {
        "health_mom": _fspec(transition_function="linear"),
        "ln_inv": _fspec(
            is_endogenous=False,
            transition_function="linear",
            correction=CorrectionSpec(instruments=("sum_inv_paid_log",)),
        ),
    }
    model = ModelSpec(factors=factors, observed_factors=("sum_inv_paid_log",))
    with pytest.raises(ValueError, match="endogenous"):
        _resolve_control_function(model)


def test_augmented_factor_spec_forwards_optional_flags() -> None:
    """Augmentation must propagate every `FactorSpec` flag, not just the obvious ones.

    Regression: the augmented `FactorSpec` constructor previously omitted
    `has_production_shock` and `has_initial_distribution`, both of which
    default to `True`. Any model that set either flag to `False` saw the
    flag silently reset to `True` once endogenous-period augmentation ran,
    producing a different model than the user specified.
    """
    from skillmodels.common.process_model import (  # noqa: PLC0415
        _augment_periods_for_endogenous_factors,
        _get_labels,
        get_dimensions,
    )

    fac3 = MODEL2.factors["fac3"]
    corr = CorrectionSpec(instruments=("inv_z",))
    custom_fac3 = replace(
        fac3,
        is_endogenous=True,
        has_production_shock=False,
        has_initial_distribution=False,
        correction=corr,
    )
    new_factors = dict(MODEL2.factors) | {"fac3": custom_fac3}
    model = MODEL2._replace(factors=new_factors)._replace(stagemap=None)

    dims = get_dimensions(model_spec=model, has_endogenous_factors=True)
    labels = _get_labels(model_spec=model, has_endogenous_factors=True, dimensions=dims)
    aug_spec = _augment_periods_for_endogenous_factors(
        model_spec=model, dimensions=dims, labels=labels
    )
    aug_fac3 = aug_spec.factors["fac3"]
    assert aug_fac3.has_production_shock is False
    assert aug_fac3.has_initial_distribution is False
    # Sanity: the explicitly-set is_endogenous survives too.
    assert aug_fac3.is_endogenous is True
    # The control-function correction must survive augmentation too.
    assert aug_fac3.correction is corr
