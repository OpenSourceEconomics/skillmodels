"""Tests for the reusable transition-aware identification anchor check (audit F6).

`check_identification` is the estimator-agnostic core of the AF period-0 anchor
diagnostic: given a ModelSpec plus any fixed_params / constraints, it returns a
list of human-readable problems with the initial-period affine anchoring. Every
factor with an initial distribution needs both a loading (scale) and an intercept
(location) anchor at period 0; the CES simplex does NOT supply the initial
location anchor (Pro F1). It is exposed for CHS/common tooling but is deliberately
NOT wired into the default process_model path (no pipeline-wide gating).
"""

import functools
from collections.abc import Mapping

import optimagic as om
import pandas as pd

from skillmodels.common.constraints import select_by_loc
from skillmodels.common.identification import check_identification
from skillmodels.common.model_spec import (
    FactorSpec,
    ModelSpec,
    Normalizations,
)


def _model(
    *,
    skills_loadings: tuple[Mapping[str, float], ...],
    skills_intercepts: tuple[Mapping[str, float], ...],
    skills_transition: str = "translog",
) -> ModelSpec:
    return ModelSpec(
        factors={
            "skills": FactorSpec(
                measurements=(("y1", "y2", "y3"),) * 2,
                normalizations=Normalizations(
                    loadings=skills_loadings,
                    intercepts=skills_intercepts,
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
    )


def test_check_identification_clean_translog_returns_no_problems() -> None:
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({"y1": 0},) * 2,
    )
    assert check_identification(model) == []


def test_check_identification_flags_missing_period0_loading() -> None:
    model = _model(
        skills_loadings=({}, {"y1": 1}),
        skills_intercepts=({"y1": 0},) * 2,
    )
    problems = check_identification(model)
    assert any("period 0" in p for p in problems)


def test_check_identification_log_ces_requires_initial_intercept_anchor() -> None:
    # The CES simplex replaces only the CROSS-PERIOD location alternative, not
    # the absolute initial anchor mu_theta,0,1=0 (Pro F1: f(x+c,i+c)=f(x,i)+c
    # leaves a common-shift orbit). So plain log_ces still needs a period-0
    # intercept anchor.
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {}),
        skills_transition="log_ces",
    )
    problems = check_identification(model)
    assert any("period 0" in p and "location" in p for p in problems)


def test_check_identification_equality_of_free_loadings_does_not_anchor() -> None:
    # An equality among free loadings reduces dimension but leaves their common
    # scale free (Pro F3): it is NOT an anchor unless connected to a fixed or
    # normalized member. period-0 skills loading is empty and the group ties two
    # otherwise-free loadings, so the scale stays unanchored -> a problem.
    model = _model(
        skills_loadings=({}, {}),
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
        om.EqualityConstraint(selector=functools.partial(select_by_loc, loc=group))
    ]
    problems = check_identification(model, constraints=constraints)
    assert any("period 0" in p and "scale" in p for p in problems)


def test_check_identification_honours_fixed_params_anchor() -> None:
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
    assert check_identification(model, fixed_params=fixed) == []
