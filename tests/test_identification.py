"""Tests for the reusable transition-aware identification anchor check (audit F6).

`check_identification` is the estimator-agnostic core of the AF period-0 anchor
diagnostic: given a ModelSpec plus any fixed_params / constraints, it returns a
list of human-readable problems with the initial-period affine anchoring,
dispatching on the transition type (trans-log needs a loading + intercept
anchor; plain log_ces needs only the loading anchor since the simplex supplies
the location). It is exposed for CHS/common tooling but is deliberately NOT
wired into the default process_model path (no pipeline-wide gating).
"""

from collections.abc import Mapping

import pandas as pd

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


def test_check_identification_log_ces_needs_no_intercept_anchor() -> None:
    model = _model(
        skills_loadings=({"y1": 1},) * 2,
        skills_intercepts=({}, {}),
        skills_transition="log_ces",
    )
    assert check_identification(model) == []


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
