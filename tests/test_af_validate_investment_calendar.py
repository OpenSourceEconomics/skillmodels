"""Regression tests for the AF endogenous-investment measurement-calendar guard.

AF reconstructs an endogenous factor (investment) from the PREVIOUS period's
latent skills: at the (t-1)->t step the generated investment I is a function of
theta_{t-1}, and it is that same I which the period-t measurement block scores.
So measurements declared at period t for an endogenous factor measure the
investment generated from period t-1 (I_{t-1}), not the contemporaneous I_t.
CHS / AMN read the identical period-t measurements as the contemporaneous I_t (a
standard latent factor with its own initial distribution). The two are different
calendars for one ModelSpec, so `validate_af_model` warns when an endogenous
factor carries measurements (audit F8). Modelling investment as a standard
non-endogenous factor instead measures the contemporaneous value and emits no
calendar warning.
"""

import warnings

import pytest

from skillmodels.af.validate import validate_af_model
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _skills_factor() -> FactorSpec:
    return FactorSpec(
        measurements=(("y1", "y2", "y3"),) * 2,
        normalizations=Normalizations(
            loadings=({"y1": 1},) * 2,
            intercepts=({"y1": 0},) * 2,
        ),
        transition_function="translog_af",
    )


def test_validate_af_model_warns_on_endogenous_factor_with_measurements() -> None:
    """An endogenous factor with measurements triggers the calendar warning.

    Its period-1 indicators score the investment generated from period-0 skills
    (I_0), not the contemporaneous I_1, so the user is warned that AF and CHS/AMN
    read this ModelSpec on different investment calendars.
    """
    model = ModelSpec(
        factors={
            "skills": _skills_factor(),
            "investment": FactorSpec(
                measurements=((), ("z1", "z2", "z3")),
                normalizations=Normalizations(
                    loadings=({}, {"z1": 1}),
                    intercepts=({}, {"z1": 0}),
                ),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
    )
    with pytest.warns(UserWarning, match="generated from period"):
        validate_af_model(model)


def test_validate_af_model_no_calendar_warning_for_standard_investment() -> None:
    """A standard (non-endogenous) investment factor measures the contemporaneous value.

    With `is_endogenous=False` the factor is an ordinary latent state factor whose
    period-t measurements score I_t -- the same calendar as CHS/AMN -- so no
    calendar warning fires.
    """
    model = ModelSpec(
        factors={
            "skills": _skills_factor(),
            "investment": FactorSpec(
                measurements=(("z1", "z2", "z3"),) * 2,
                normalizations=Normalizations(
                    loadings=({"z1": 1},) * 2,
                    intercepts=({"z1": 0},) * 2,
                ),
                transition_function="linear",
            ),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model) is None


def test_validate_af_model_no_calendar_warning_for_measurementless_endogenous() -> None:
    """An endogenous factor with no measurements has nothing to mis-calendar."""
    model = ModelSpec(
        factors={
            "skills": _skills_factor(),
            "investment": FactorSpec(
                measurements=((), ()),
                normalizations=Normalizations(loadings=({}, {}), intercepts=({}, {})),
                transition_function="linear",
                is_endogenous=True,
                has_initial_distribution=False,
            ),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model) is None
