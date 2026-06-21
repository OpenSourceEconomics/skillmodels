"""Regression tests for the AF period-0 normalization-anchor check (audit F7b).

`validate_af_model` previously rejected only `normalizations is None`. A
`Normalizations` object whose period-0 maps are empty therefore slipped
through, leaving the initial factor distribution's affine orbit (scale +
location) unpinned and the trans-log model under-identified.

The period-0 distribution is not produced by any transition, so its anchor
must be supplied directly: a loading/intercept normalization, a `fixed_params`
pin, or an equality constraint tying it to an anchored parameter. These tests
pin that period-0 check. Periods t>0 (where the transition can legitimately
propagate the anchor) are deliberately out of scope here -- the
transition-aware identification check is a separate (P4) item.
"""

import functools
import warnings
from collections.abc import Mapping

import optimagic as om
import pandas as pd
import pytest

from skillmodels.af.validate import validate_af_model
from skillmodels.common.constraints import select_by_loc
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _model(
    *,
    skills_loadings: tuple[Mapping[str, float], ...],
    skills_intercepts: tuple[Mapping[str, float], ...],
) -> ModelSpec:
    """Build a two-period AF model with controllable skills normalizations."""
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=skills_loadings,
                    intercepts=skills_intercepts,
                ),
                transition_function="translog_af",
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
    )


def test_validate_af_model_raises_on_empty_period0_loading() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    with pytest.raises(ValueError, match="period 0"):
        validate_af_model(model)


def test_validate_af_model_raises_on_empty_period0_intercept() -> None:
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {"y1": 0}),
    )
    with pytest.raises(ValueError, match="period 0"):
        validate_af_model(model)


def test_validate_af_model_raises_on_all_empty_normalizations() -> None:
    model = _model(
        skills_loadings=({}, {}),
        skills_intercepts=({}, {}),
    )
    with pytest.raises(ValueError, match="period 0"):
        validate_af_model(model)


def test_validate_af_model_accepts_period0_loading_via_fixed_params() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    fixed = pd.DataFrame(
        {"value": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("loadings", 0, "y1", "skills")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model, fixed_params=fixed) is None


def test_validate_af_model_accepts_period0_intercept_via_fixed_params() -> None:
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {"y1": 0}),
    )
    fixed = pd.DataFrame(
        {"value": [0.0]},
        index=pd.MultiIndex.from_tuples(
            [("controls", 0, "y1", "constant")],
            names=["category", "period", "name1", "name2"],
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model, fixed_params=fixed) is None


def test_validate_af_model_accepts_period0_loading_via_equality_constraint() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    group = pd.MultiIndex.from_tuples(
        [
            ("loadings", 0, "y1", "skills"),
            ("loadings", 1, "y1", "skills"),
        ],
        names=["category", "period", "name1", "name2"],
    )
    constraints: list[om.constraints.Constraint] = [
        om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=group)),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model, constraints=constraints) is None


def test_validate_af_model_accepts_empty_normalization_at_later_period() -> None:
    # Period 0 is anchored; period 1 maps are empty. The transition can carry
    # the anchor forward, so this must NOT raise (transition-aware check is P4).
    model = _model(
        skills_loadings=({"y1": 1}, {}),
        skills_intercepts=({"y1": 0}, {}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert validate_af_model(model) is None
