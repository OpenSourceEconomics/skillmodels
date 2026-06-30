"""Regression tests for the AF observed-factor production-leakage guard.

These pin two AF-F3 fixes:

* the production factor using a built-in transition function that enumerates
  parameters over `all_factors` (latent + observed) while observed factors
  (e.g. income) are present triggers a loud `UserWarning`, because the
  observed factors would otherwise enter the production function with free
  coefficients (a silent wrong-estimand). The warning is non-breaking by
  design (see deviation note in the implementation); and
* the paper-matching `translog_af` / `log_ces_af` variants (which only
  enumerate production factors) are accepted without a warning.
"""

import warnings

import pytest

from skillmodels.af.validate import validate_af_model
from skillmodels.common.model_spec import (
    CorrectionSpec,
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _build_model(
    *,
    skills_transition: str,
    observed_factors: tuple[str, ...],
) -> ModelSpec:
    """Build a minimal AF model: one production factor + one investment factor."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"y1": 1},) * 2,
                    intercepts=({"y1": 0},) * 2,
                ),
                transition_function=skills_transition,
            ),
            "investment": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"z1": 1},) * 2,
                    intercepts=({"z1": 0},) * 2,
                ),
                transition_function="linear",
                is_endogenous=True,
            ),
        },
        observed_factors=observed_factors,
    )


def test_validate_af_model_warns_on_observed_factor_in_builtin_production() -> None:
    model = _build_model(
        skills_transition="translog",
        observed_factors=("income",),
    )
    with pytest.warns(UserWarning, match="only through the investment equation"):
        validate_af_model(model)


def test_validate_af_model_no_warning_when_no_observed_factors() -> None:
    model = _build_model(
        skills_transition="translog",
        observed_factors=(),
    )
    # No observed factors means no leakage is possible: no leakage warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model) is None


def test_validate_af_model_no_warning_with_translog_af() -> None:
    model = _build_model(
        skills_transition="translog_af",
        observed_factors=("income",),
    )
    # `translog_af` enumerates production factors only -> not leaky, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model) is None


def test_validate_af_model_raises_on_correction_spec() -> None:
    # AF implements only the kappa=0 (exogenous-investment) special case, so a
    # CorrectionSpec must raise loudly rather than silently estimate a different
    # estimand than CHS on the identical spec.
    model = _build_model(
        skills_transition="translog_af",
        observed_factors=("income",),
    )
    investment = FactorSpec(
        measurements=(("z1", "z2", "z3"),) * 2,
        normalizations=Normalizations(
            loadings=({"z1": 1},) * 2,
            intercepts=({"z1": 0},) * 2,
        ),
        transition_function="linear",
        is_endogenous=True,
        correction=CorrectionSpec(instruments=("income",)),
    )
    model = ModelSpec(
        factors={"skills": model.factors["skills"], "investment": investment},
        observed_factors=("income",),
    )
    with pytest.raises(NotImplementedError, match="control-function correction"):
        validate_af_model(model)
