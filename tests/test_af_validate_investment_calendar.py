"""Regression tests for the AF endogenous-investment measurement calendar.

On the calendar-adapter path a reconstructed endogenous factor (investment,
`is_endogenous=True` and `has_initial_distribution=False`) is contemporaneous: a
period-c investment indicator measures `I_c`, the same calendar CHS / AMN read.
The adapter makes the public ModelSpec contemporaneous and the AF step assembler
re-times the source investment internally, so `validate_af_model` must NOT warn
that the indicators denote `I_{t-1}` or that estimators disagree on the calendar.
A user who shifted their data to satisfy such a warning would reintroduce the
original off-by-one bug.
"""

import warnings

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


def _lagged_calendar_warned(records: list[warnings.WarningMessage]) -> bool:
    """Return whether any recorded warning teaches the stale lagged calendar."""
    return any(
        "I_{t-1}" in str(r.message)
        or "different investment calendars" in str(r.message)
        or "generated from period" in str(r.message)
        for r in records
    )


def test_no_lagged_calendar_warning_for_reconstructed_endogenous() -> None:
    """A reconstructed endogenous factor with measurements emits no calendar warning.

    Under the calendar adapter its period-c indicators score the contemporaneous
    `I_c` (shared with CHS / AMN), so `validate_af_model` must not claim they
    denote `I_{t-1}` or that estimators read different investment calendars.
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
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        validate_af_model(model)
    assert not _lagged_calendar_warned(records)


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
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        validate_af_model(model)
    assert not _lagged_calendar_warned(records)


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
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        validate_af_model(model)
    assert not _lagged_calendar_warned(records)
